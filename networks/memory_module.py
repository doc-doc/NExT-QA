import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init


class AttentionGRUCell(nn.Module):
    '''
    Eq (1)~(4), then modify by Eq (11)
    When forwarding, we feed attention gate g into GRU
    '''
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.W = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
    
    def init_weights(self):
        self.Wr.weight.data.normal_(0.0, 0.02)
        self.Wr.bias.data.fill_(0)
        self.Ur.weight.data.normal_(0.0, 0.02)
        self.Ur.bias.data.fill_(0)        
        self.W.weight.data.normal_(0.0, 0.02)
        self.W.bias.data.fill_(0)        
        self.U.weight.data.normal_(0.0, 0.02)
        self.U.bias.data.fill_(0)

    def forward(self, fact, C, g):
        '''
        fact.size() -> (#batch, #hidden = #embedding)
        c.size() -> (#hidden, ) -> (#batch, #hidden = #embedding)
        r.size() -> (#batch, #hidden = #embedding)
        h_tilda.size() -> (#batch, #hidden = #embedding)
        g.size() -> (#batch, )
        '''

        r = torch.sigmoid(self.Wr(fact) + self.Ur(C))
        h_tilda = torch.tanh(self.W(fact) + r * self.U(C))
        g = g.unsqueeze(1).expand_as(h_tilda)
        h = g * h_tilda + (1 - g) * C
        return h

class AttentionGRU(nn.Module):
    '''
    Section 3.3
    continuously run AttnGRU to get contextual vector c at each time t
    '''
    def __init__(self, input_size, hidden_size):
        super(AttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.AGRUCell = AttentionGRUCell(input_size, hidden_size)
    
    def init_weights(self):
        self.AGRUCell.init_weights()
        
    def forward(self, facts, G):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        fact.size() -> (#batch, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        g.size() -> (#batch, )
        C.size() -> (#batch, #hidden)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        C = Variable(torch.zeros(self.hidden_size)).cuda()
        for sid in range(sen_num):
            fact = facts[:, sid, :]
            g = G[:, sid]
            if sid == 0:
                C = C.unsqueeze(0).expand_as(fact)
            C = self.AGRUCell(fact, C, g)
        return C
                
class EpisodicMemory(nn.Module):
    '''
    Section 3.3
    '''

    def __init__(self, hidden_size):
        super(EpisodicMemory, self).__init__()
        self.AGRU = AttentionGRU(hidden_size, hidden_size)
        self.z1 = nn.Linear(4 * hidden_size, hidden_size)
        self.z2 = nn.Linear(hidden_size, 1)
        self.next_mem = nn.Linear(3 * hidden_size, hidden_size)
        
        
    def init_weights(self):
        self.z1.weight.data.normal_(0.0, 0.02)
        self.z1.bias.data.fill_(0)
        self.z2.weight.data.normal_(0.0, 0.02)
        self.z2.bias.data.fill_(0)
        self.next_mem.weight.data.normal_(0.0, 0.02)
        self.next_mem.bias.data.fill_(0)
        self.AGRU.init_weights()
        
        
    def make_interaction(self, frames, questions, prevM):
        '''
        frames.size() -> (#batch, T, #hidden = #embedding)
        questions.size() -> (#batch, 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        z.size() -> (#batch, T, 4 x #embedding)
        G.size() -> (#batch, T)
        '''
        batch_num, T, embedding_size = frames.size()
        questions = questions.view(questions.size(0),1,questions.size(1))
        
        
        #questions = questions.expand_as(frames)
        #prevM = prevM.expand_as(frames)
    
        #print(questions.size(),prevM.size())
        
        # Eq (8)~(10)
        z = torch.cat([
            frames * questions,
            frames * prevM,
            torch.abs(frames - questions),
            torch.abs(frames - prevM)
        ], dim=2)

        z = z.view(-1, 4 * embedding_size)

        G = torch.tanh(self.z1(z))
        G = self.z2(G)
        G = G.view(batch_num, -1)
        G = F.softmax(G,dim=1)
        #print('G size',G.size())
        return G

    def forward(self, frames, questions, prevM):
        '''
        frames.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #sentence = 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        C.size() -> (#batch, #hidden)
        concat.size() -> (#batch, 3 x #embedding)
        '''

        '''
        section 3.3 - Attention based GRU
        input: F and q, as frames and questions
        then get gates g
        then (c,m,g) feed into memory update module Eq(13)
        output new memory state
        '''
        # print(frames.shape, questions.shape, prevM.shape)
        
        G = self.make_interaction(frames, questions, prevM)
        C = self.AGRU(frames, G)
        concat = torch.cat([prevM.squeeze(1), C, questions.squeeze(1)], dim=1)
        next_mem = F.relu(self.next_mem(concat))
        #print(next_mem.size())
        next_mem = next_mem.unsqueeze(1)
        return next_mem
