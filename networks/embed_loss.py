import torch
import torch.nn as nn
import torch.nn.functional as F

# __all__ = ['MultipleChoiceLoss', 'CountLoss']


class MultipleChoiceLoss(nn.Module):

    def __init__(self, num_option=5, margin=1, size_average=True):
        super(MultipleChoiceLoss, self).__init__()
        self.margin = margin
        self.num_option = num_option
        self.size_average = size_average

    # score is N x C

    def forward(self, score, target):
        N = score.size(0)
        C = score.size(1)
        assert self.num_option == C

        loss = torch.tensor(0.0).cuda()
        zero = torch.tensor(0.0).cuda()

        cnt = 0
        #print(N,C)
        for b in range(N):
            # loop over incorrect answer, check if correct answer's score larger than a margin
            c0 = target[b]
            for c in range(C):
                if c == c0:
                    continue

                # right class and wrong class should have score difference larger than a margin
                # see formula under paper Eq(4)
                loss += torch.max(zero, 1.0 + score[b, c] - score[b, c0])
                cnt += 1

        if cnt == 0:
            return loss

        return loss / cnt if self.size_average else loss
