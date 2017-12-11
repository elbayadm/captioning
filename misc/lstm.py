import torch
from torch import nn
from torch.nn import functional as F


def step_attention(query, attention):
    """
    Input:
        query: rnn hidden state (N, D)
        attention: region embeddings (N, D, W, H)
    Outpt:
        context : weighted sum of attended Regions
        attn_scores : weight assigned to each
    """
    bs, d, w, h = attention.size()
    attention = attention.contiguous().view(bs, d, -1)
    # print("Regions :", attention.size())
    # print("Query:", query.unsqueeze(1).size())
    score = torch.matmul(query.unsqueeze(1), attention).squeeze(1)
    # print('Scores:', score.size())
    normalized_score = F.softmax(score)
    # print('Attention scores:', normalized_score[0])
    # context = sum e_i a_i
    context = torch.matmul(normalized_score.unsqueeze(1),
                           attention.transpose(1, 2)).squeeze(1)
    # print('context:', context.size())
    return context, normalized_score.contiguous().view(bs, w, h)


class MultiLayerLSTMCells(nn.Module):
    """ stack multiple LSTM Cells"""
    def __init__(self, input_size, hidden_size, num_layers,
                 bias=True, dropout=0.0):
        super().__init__()
        cells = []
        cells.append(nn.LSTMCell(input_size, hidden_size, bias))
        for _ in range(num_layers-1):
            cells.append(nn.LSTMCell(hidden_size, hidden_size, bias))
        self._cells = nn.ModuleList(cells)
        self._dropout = dropout
        self.reset_parameters()

    def forward(self, input_, state):
        """
        Arguments:
            input_: Variable of FloatTensor (batch, input_size)
            states: tuple of the H, C LSTM states
                Variable of FloatTensor (num_layers, batch, hidden_size)
        Returns:
            LSTM states
            new_h: (num_layers, batch, hidden_size)
            new_c: (num_layers, batch, hidden_size)
        """
        hs = []
        cs = []
        for i, cell in enumerate(self._cells):
            s = (state[0][i, :, :], state[1][i, :, :])
            h, c = cell(input_, s)
            hs.append(h)
            cs.append(c)
            input_ = F.dropout(h, p=self._dropout, training=self.training)

        new_h = torch.stack(hs, dim=0)
        new_c = torch.stack(cs, dim=0)

        return new_h, new_c

    def reset_parameters(self):
        for cell in self._cells:
            # xavier initialization
            gate_size = self.hidden_size/4
            for weight in [cell.weight_ih, cell.weight_hh]:
                for w in torch.chunk(weight.data, 4, dim=0):
                    nn.init.xavier_normal(w)
            # forget_bias = 1
            for bias in [cell.bias_ih, cell.bias_hh]:
                torch.chunk(bias.data, 4, dim=0)[1].fill_(1)

    @property
    def hidden_size(self):
        return self._cells[0].hidden_size

    @property
    def input_size(self):
        return self._cells[0].input_size

    @property
    def num_layers(self):
        return len(self._cells)

    @property
    def bidirectional(self):
        return False


class LSTMAttnDecoder(nn.Module):
    def __init__(self, embedding, lstm_cell, attention, projection, logit):
        """
        inputs:
            E = word embedding size, D = rnn hidden size
            embedding: Word embedding layer (a lookup table) V x E
            lstm_cell: RNN LSMT cell dim(x) = E + D, dim(h) = D
            attention: Linear layer D x D
            projection: Linear layer (E + 2D) x D
            logit:      Linear layer D x V
        """
        super().__init__()
        self._embedding = embedding
        self._lstm_cell = lstm_cell
        self._attention = attention
        self._projection = projection
        self._logit = logit

    def forward(self, enc_img, input, init_states):
        """
        enc_img     : region embeddings a_1,...a_L
        input       : gt sequence of tokens
        inti_states : h_0, c_0 (obtained as MLP(avearge(a_i)))T
        """
        batch_size, max_len = input.size()
        logits = []
        states = init_states
        for i in range(max_len):
            tok = input[:, i:i+1]
            logit, states, _ = self._step(tok, states, enc_img)
            logits.append(logit)
        return torch.stack(logits, dim=1)

    def _step(self, tok, states, attention):
        h, c = states
        # print('initial states:', h.size(), c.size(), h[-1].size())
        context, attn_scores = step_attention(self._attention(h[-1]), attention)
        emb = self._embedding(tok).squeeze(1)
        x = torch.cat([emb, context], dim=1).unsqueeze(0)
        # print('Feeding to the RNN:', x.size())
        out, (h, c) = self._lstm_cell(x, (h, c))
        # print('LSTM outputs:', out.size(), h.size(), c.size(), out[-1].size())
        input_proj = torch.cat([emb, context, out[-1]], dim=1)
        output = self._projection(input_proj)
        # print('Final output size:', output.size())
        # logit = F.log_softmax(torch.mm(output, self._embedding.weight.t()))  # If using a single embedding matrix
        logit = F.log_softmax(self._logit(output))
        return logit, (h, c), attn_scores

    def decode_step(self, tok, states, attention):
        logit, states, attn = self._step(tok, states, attention)
        prob, out = logit.max(dim=1, keepdim=True)
        return out, prob, states, attn

