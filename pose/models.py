import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def init_lstm(l):
    # positive forget gate bias (Jozefowicz et al., 2015)
    for names in l._all_weights:
        for name in filter(lambda n: "bias" in n, names):
            bias = getattr(l, name)
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.)


class LSTMSkeleton(nn.Module):
    def __init__(self, num_input, args=None):
        super().__init__()
        self.args = args

        self.encoder = nn.Sequential(
            nn.Linear(num_input, args.num_input_lstm),
            nn.Dropout(args.dropout),
            nn.ReLU(True)
        )

        self.lstm = nn.LSTM(args.num_input_lstm, args.hidden_size, batch_first=True, num_layers=args.num_layers,
                            bidirectional=args.bidirectional, dropout=args.dropout)

        init_lstm(self.lstm)

    def forward(self, features, lengths):
        if self.args.bidirectional:
            num_directions = 2
        else:
            num_directions = 1

        confidences = features[:, :, :, 2]
        features_positions_x = features[:, :, :, 0].clone()
        features_positions_y = features[:, :, :, 1].clone()

        t = torch.tensor([self.args.confidence_threshold]).cuda()  # threshold
        confidences = (confidences > t).float() * 1
        features_positions = torch.stack(
            (features_positions_x * confidences, features_positions_y * confidences), dim=3)

        features = features_positions.view(features_positions.size(0), features_positions.size(1), -1)

        encoded_features = self.encoder(features)

        h0 = torch.zeros(self.args.num_layers * num_directions, encoded_features.size(0),
                         self.args.hidden_size).cuda()  # 2 for bidirection
        c0 = torch.zeros(self.args.num_layers * num_directions, encoded_features.size(0), self.args.hidden_size).cuda()

        packed = pack_padded_sequence(encoded_features, lengths, batch_first=True)

        output, (hn, cn) = self.lstm(packed, (h0, c0))
        output, _ = pad_packed_sequence(output, batch_first=True)

        if self.args.lstm_pooling == "sum":
            output = torch.sum(output, dim=1) / lengths.unsqueeze(1).float()
        elif self.args.lstm_pooling == "max":
            output, _ = torch.max(output, dim=1)
        elif self.args.lstm_pooling == "last":
            output = output[torch.arange(output.size(0)), lengths - 1, :]
        elif self.args.lstm_pooling == "sum+hidden":
            output = torch.sum(output, dim=1) / lengths.unsqueeze(1).float()
            hiddens = hn.permute(1, 0, 2).contiguous()
            output = torch.cat((output, hiddens.view(hiddens.size(0), -1)), dim=1)

        return output


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class BodyFaceEmotionClassifier(nn.Module):
    def __init__(self, args):
        super(BodyFaceEmotionClassifier, self).__init__()
        self.args = args

        """ use simple dnn for modeling the skeleton """
        # this is the number of openpose skeleton joints: 21 2D points for hands and 25 2D points for body
        n = 42 + 42 + 50
        self.static = nn.Sequential(
            nn.Linear(n, args.first_layer_size),
            nn.ReLU()
        )
        self.bn_body = nn.BatchNorm1d(args.first_layer_size)
        self.classifier_body = nn.Sequential(
            nn.Linear(args.first_layer_size, args.num_classes)
        )

    def forward(self, inp):
        body, hand_right, hand_left, length = inp

        features = torch.cat((body, hand_right, hand_left), dim=2)
        features = features.view(features.size(0), features.size(1), -1, 3)

        features_positions_x = features[:, :, :, 0].clone()
        features_positions_y = features[:, :, :, 1].clone()

        confidences = features[:, :, :, 2].clone()
        t = torch.tensor([self.args.confidence_threshold]).cuda()  # threshold for confidence of joints
        confidences = (confidences > t).float() * 1

        # make all joints with threshold lower than
        features_positions = torch.stack(
            (features_positions_x * confidences, features_positions_y * confidences), dim=3
        )

        static_features = features_positions.view(features_positions.size(0), features_positions.size(1), -1)
        static_features = self.static(static_features)

        # feats.append(torch.max(static_features,dim=1)/length.unsqueeze(1).float())

        sum_ = torch.zeros(body.size(0), static_features.size(2)).float().cuda()
        if self.args.body_pooling == "max":
            for i in range(0, body.size(0)):
                sum_[i] = torch.max(static_features[i, :length[i], :], dim=0)[0]
        elif self.args.body_pooling == "avg":
            for i in range(0, body.size(0)):
                sum_[i] = torch.sum(static_features[i, :length[i], :], dim=0) / length[i].float()

        out_body = self.classifier_body(
            self.bn_body(torch.sum(static_features, dim=1) / length.unsqueeze(1).float())
        )

        return out_body
