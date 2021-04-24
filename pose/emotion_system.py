from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as data
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from datasets import *
from models import *


class EmotionRecognitionSystem:
    def __init__(self, args):
        print(args)
        self.args = args

    def run(self):
        args = self.args

        # array to store accuracies from all 10 iterations
        self.all_iteration_accuracies = []

        self.confusion_matrix = np.empty(0)

        all_iterations_accuracy_meter_top_all = []
        all_iterations_accuracy_meter_top_face = []
        all_iterations_accuracy_meter_top_body = []
        all_iterations_p = []
        all_iterations_r = []
        all_iterations_f = []

        for i in range(self.args.num_total_iterations):
            self.current_iteration = i
            val_top_all, val_top_body, val_top_face, p, r, f = self.cross_validation(num_splits=args.num_splits)

            all_iterations_accuracy_meter_top_all.append(val_top_all)
            all_iterations_accuracy_meter_top_face.append(val_top_face)
            all_iterations_accuracy_meter_top_body.append(val_top_body)
            all_iterations_p.append(p)
            all_iterations_r.append(r)
            all_iterations_f.append(f)

            print(
                f'[Iteration: {i + 1:02d}/{args.num_total_iterations:02d}] '
                f'Top1 Accuracy: {np.mean(all_iterations_accuracy_meter_top_all):.3f} '
                f'Accuracy Body {np.mean(all_iterations_accuracy_meter_top_body):.3f} '
                f'Accuracy Face {np.mean(all_iterations_accuracy_meter_top_face):.3f} '
                f'SK Prec: {np.mean(all_iterations_p):.3f}, SK Rec: {np.mean(all_iterations_r):.3f} '
                f'F-Score: {np.mean(all_iterations_f):.3f}'
            )

    def get_scaler(self):
        scaler = {}
        feats = ["bodies", "faces", "hands_right", "hands_left", ]

        acc1 = 0
        acc2 = 0

        for x in feats:
            all_data = np.vstack(getattr(self.train_dataset, x))

            scaler[x] = MinMaxScaler()
            scaler[x].fit(all_data)

        return scaler

    def cross_validation(self, num_splits):
        cross_val_accuracy_meter_top_all = []
        cross_val_accuracy_meter_top_face = []
        cross_val_accuracy_meter_top_body = []

        cross_val_p = []
        cross_val_r = []
        cross_val_f = []

        data = get_babyrobot_data()
        faces, bodies, hands_right, hands_left, lengths, Y, Y_face, Y_body, paths, groups = data

        self.kfold = GroupKFold(n_splits=num_splits).split(bodies, Y_body, groups)

        for n in range(num_splits):
            self.current_split = n
            data = get_babyrobot_data()
            train_idx, test_idx = next(self.kfold)
            self.train_dataset = BodyFaceDataset(data=data, indices=train_idx, phase="train", args=self.args)
            self.test_dataset = BodyFaceDataset(data=data, indices=test_idx, phase="val", args=self.args)

            print("train samples: %d" % len(self.train_dataset))
            print(np.bincount(self.train_dataset.Y))

            print(np.bincount(self.train_dataset.Y_face))
            print(np.bincount(self.train_dataset.Y_body))

            print("test samples: %d" % len(self.test_dataset))
            print(np.bincount(self.test_dataset.Y))

            print(np.bincount(self.test_dataset.Y_face))
            print(np.bincount(self.test_dataset.Y_body))

            scaler = self.get_scaler()

            self.train_dataset.set_scaler(scaler)
            self.test_dataset.set_scaler(scaler)

            self.train_dataset.to_tensors()
            self.test_dataset.to_tensors()

            self.train_dataset.prepad()
            self.test_dataset.prepad()

            print("scaled data")

            if self.args.batch_size == -1:
                batch_size = len(self.train_dataset)
            else:
                batch_size = self.args.batch_size

            self.dataloader_train = torch.utils.data.DataLoader(
                self.train_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=4
            )
            self.dataloader_test = torch.utils.data.DataLoader(
                self.test_dataset, batch_size=len(self.test_dataset), num_workers=4
            )

            self.model = BodyFaceEmotionClassifier(self.args).cuda()

            start = time.time()

            val_top_all, val_top_body, val_top_face, p, r, f = self.fit(self.model)

            end = time.time()

            cross_val_accuracy_meter_top_all.append(val_top_all)
            cross_val_accuracy_meter_top_body.append(val_top_body)
            cross_val_accuracy_meter_top_face.append(val_top_face)

            cross_val_p.append(p)
            cross_val_r.append(r)
            cross_val_f.append(f)
            print(val_top_all, val_top_body, val_top_face, p, r, f)
            print(
                f'[Split: {n + 1:02d}/{num_splits:02d}] Accuracy: {np.mean(cross_val_accuracy_meter_top_all):.3f} '
                f'Body Accuracy: {np.mean(cross_val_accuracy_meter_top_body):.3f} '
                f'Face Accuracy: {np.mean(cross_val_accuracy_meter_top_face):.3f} '
                f'SK Prec: {np.mean(cross_val_p):.3f} SK Rec: {np.mean(cross_val_r):.3f} '
                f'F-Score: {np.mean(cross_val_f):.3f} Time: {end - start:.3f}'
            )

        return np.mean(cross_val_accuracy_meter_top_all), np.mean(cross_val_accuracy_meter_top_body), np.mean(
            cross_val_accuracy_meter_top_face), \
               np.mean(cross_val_p), np.mean(cross_val_r), np.mean(cross_val_f)

    def fit(self, model):
        if self.args.weighted_loss:
            if self.args.split_branches:
                self.criterion_both = nn.CrossEntropyLoss().cuda()
                self.criterion_face = nn.CrossEntropyLoss().cuda()
                self.criterion_body = nn.CrossEntropyLoss(
                    weight=torch.FloatTensor(get_weighted_loss_weights(self.train_dataset.Y_body, 7))).cuda()
            elif self.args.use_labels == "body":
                self.criterion = nn.CrossEntropyLoss(
                    weight=torch.FloatTensor(get_weighted_loss_weights(self.train_dataset.Y_body, 7))).cuda()
            else:
                self.criterion = nn.CrossEntropyLoss().cuda()
        else:
            if self.args.split_branches:
                self.criterion_both = nn.CrossEntropyLoss().cuda()
                self.criterion_face = nn.CrossEntropyLoss().cuda()
                self.criterion_body = nn.CrossEntropyLoss().cuda()
            elif self.args.use_labels == "body":
                self.criterion = nn.CrossEntropyLoss().cuda()
            else:
                self.criterion = nn.CrossEntropyLoss().cuda()

        if self.args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                              weight_decay=self.args.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                             weight_decay=self.args.weight_decay, momentum=self.args.momentum)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=0.1)

        best_acc = 0

        self.mat = np.empty(0)

        for self.current_epoch in range(0, self.args.epochs):
            train_acc, train_loss = self.train_epoch()

            if self.current_epoch == self.args.epochs - 1:
                val_top_all, val_top_body, val_top_face, val_loss, p, r, f = self.eval()

            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']

            if self.current_epoch == self.args.epochs - 1:
                print(
                    f'[Epoch: {self.current_epoch:3d}/{self.args.epochs:3d}] Training Loss: {train_loss:.3f}, '
                    f'Validation Loss: {val_loss:.3f}, Training Acc: {train_acc:.3f}, Validation Acc: '
                    f'{val_top_all:.3f}, Validation Acc Body: {val_top_body:.3f}, Validation Acc Face: '
                    f'{val_top_face:.3f}, Learning Rate:{lr:.8f}')

        return val_top_all, val_top_body, val_top_face, p, r, f

    def train_epoch(self):
        self.model.train()

        accuracy_meter_top_all = AverageMeter()
        loss_meter = AverageMeter()

        all_outs_body = []
        all_outs = []
        all_outs_face = []
        all_y = []
        all_y_face = []
        all_y_body = []

        for i, batch in enumerate(self.dataloader_train):
            facial_cnn_features, face, body, hand_right, hand_left, length, y, y_face, y_body = \
                batch['facial_cnn_features'].cuda(), batch[
                    'face'].cuda(), batch['body'].cuda(), batch['hand_right'].cuda(), batch['hand_left'].cuda(), batch[
                    'length'].cuda(), batch['label'].cuda(), batch['label_face'].cuda(), batch['label_body'].cuda()

            self.optimizer.zero_grad()

            if self.args.split_branches:
                if self.args.do_fusion:
                    out, out_body, out_face, out_fusion = self.model.forward(
                        (face, body, hand_right, hand_left, length, facial_cnn_features))

                    loss = 0

                    loss_fusion = self.criterion_both(out_fusion, y)
                    loss += loss_fusion

                    if self.args.add_whole_body_branch:
                        loss_total = self.criterion_both(out, y)
                        loss += loss_total

                    loss_body = self.criterion_body(out_body, y_body)
                    loss += loss_body

                    loss_face = self.criterion_face(out_face, y_face)
                    loss += loss_face

                    # loss = loss_body + loss_face + loss_fusion

                    loss.backward()
                else:
                    out, out_body, out_face = self.model.forward(
                        (face, body, hand_right, hand_left, length, facial_cnn_features))

                    loss_total = self.criterion_both(out, y)
                    loss_body = self.criterion_body(out_body, y_body)
                    loss_face = self.criterion_face(out_face, y_face)

                    loss = loss_body + loss_face + loss_total

                    loss.backward()
            else:
                out = self.model.forward(
                    (face, body, hand_right, hand_left, length, facial_cnn_features))

                if self.args.use_labels == "body":
                    loss = self.criterion(out, y_body)
                elif self.args.use_labels == "face":
                    loss = self.criterion(out, y_face)
                else:
                    loss = self.criterion(out, y)

                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

            self.optimizer.step()
            self.scheduler.step()

            if self.current_epoch == self.args.epochs - 1:
                if self.args.split_branches:
                    if self.args.do_fusion:
                        accs = accuracy(out_fusion, y, topk=(1,))
                    else:
                        accs = accuracy(out, y, topk=(1,))
                    accuracy_meter_top_all.update(accs[0].item(), length.size(0))
                    loss_meter.update(loss.item(), length.size(0))

                else:
                    if self.args.use_labels == "body":
                        accs = accuracy(out, y_body, topk=(1,))
                    elif self.args.use_labels == "face":
                        accs = accuracy(out, y_face, topk=(1,))
                    else:
                        accs = accuracy(out, y, topk=(1,))

                    accuracy_meter_top_all.update(accs[0], body.size(0))
                    loss_meter.update(loss.item(), body.size(0))

        return accuracy_meter_top_all.avg, loss_meter.avg

    def eval(self, get_confusion_matrix=True):
        accuracy_meter_top_all = AverageMeter()
        accuracy_meter_top_face = AverageMeter()
        accuracy_meter_top_body = AverageMeter()
        loss_meter = AverageMeter()

        with torch.no_grad():
            self.model.eval()
            for i, batch in enumerate(self.dataloader_test):
                facial_cnn_features, face, body, hand_right, hand_left, length, y, y_face, y_body = \
                    batch['facial_cnn_features'].cuda(), batch[
                        'face'].cuda(), batch['body'].cuda(), batch['hand_right'].cuda(), batch['hand_left'].cuda(), \
                    batch[
                        'length'].cuda(), batch['label'].cuda(), batch['label_face'].cuda(), batch['label_body'].cuda()

                if self.args.split_branches:

                    if self.args.do_fusion:

                        out, out_body, out_face, out_fusion = self.model.forward(
                            (face, body, hand_right, hand_left, length, facial_cnn_features))

                        if not self.args.add_whole_body_branch:
                            out = out_fusion

                        accs = accuracy(out_fusion, y, topk=(1,))
                        accs_face = accuracy(out_face, y_face, topk=(1,))
                        accs_body = accuracy(out_body, y_body, topk=(1,))

                        accuracy_meter_top_all.update(accs[0].item(), length.size(0))
                        accuracy_meter_top_body.update(accs_body[0].item(), length.size(0))
                        accuracy_meter_top_face.update(accs_face[0].item(), length.size(0))

                        p, r, f, s = precision_recall_fscore_support(y.cpu(), out_fusion.detach().cpu().argmax(dim=1),
                                                                     average="macro")

                    else:
                        out, out_body, out_face = self.model.forward(
                            (face, body, hand_right, hand_left, length, facial_cnn_features))

                        accs = accuracy(out, y, topk=(1,))
                        accs_face = accuracy(out_face, y_face, topk=(1,))
                        accs_body = accuracy(out_body, y_body, topk=(1,))

                        accuracy_meter_top_all.update(accs[0].item(), length.size(0))
                        accuracy_meter_top_body.update(accs_body[0].item(), length.size(0))
                        accuracy_meter_top_face.update(accs_face[0].item(), length.size(0))

                        p, r, f, s = precision_recall_fscore_support(y.cpu(), out.detach().cpu().argmax(dim=1),
                                                                     average="macro")

                else:
                    out = self.model.forward(
                        (face, body, hand_right, hand_left, length, facial_cnn_features))

                    if self.args.use_labels == "body":
                        t = y
                        y = y_body
                    elif self.args.use_labels == "face":
                        t = y
                        y = y_face

                    accs = accuracy(out, y, topk=(1,))

                    """ change average to the desired (macro for balanced) """
                    p, r, f, s = precision_recall_fscore_support(y.cpu(), out.detach().cpu().argmax(dim=1),
                                                                 average="macro")

                    accuracy_meter_top_all.update(accs[0].item(), length.size(0))

                if self.current_epoch == self.args.epochs - 1:
                    out = out.cpu()
                    y = y.cpu()

                    np.save(
                        f"saved_scores/{self.args.exp_name}_out_{self.current_split}_{self.current_iteration:d}",
                        out
                    )
                    np.save(f"saved_scores/{self.args.exp_name}_y_{self.current_split}_{self.current_iteration:d}", y)

                    np.save(
                        f"saved_scores/{self.args.exp_name}_paths_{self.current_split}_{self.current_iteration:d}",
                        np.array(batch['paths']))

                    if self.args.split_branches:
                        out_face = out_face.cpu()
                        out_body = out_body.cpu()
                        y_face = y_face.cpu()
                        y_body = y_body.cpu()
                        np.save(
                            f"saved_scores/{self.args.exp_name}_out_face_{self.current_split}_{self.current_iteration:d}",
                            out_face)
                        np.save(
                            f"saved_scores/{self.args.exp_name}_y_face_{self.current_split}_{self.current_iteration:d}",
                            y_face)

                        np.save(
                            f"saved_scores/{self.args.exp_name}_out_body_{self.current_split}_{self.current_iteration:d}",
                            out_body)
                        np.save(
                            f"saved_scores/{self.args.exp_name}_y_body_{self.current_split}_{self.current_iteration:d}",
                            y_body)

                    if (self.args.do_fusion):
                        out_fusion = out_fusion.cpu()
                        np.save(
                            f"saved_scores/{self.args.exp_name}_out_fusion_{self.current_split}_{self.current_iteration:d}",
                            out_fusion)

                if get_confusion_matrix and self.current_epoch == self.args.epochs - 1:
                    conf = confusion_matrix(
                        y.cpu().numpy(), torch.argmax(out, dim=1).cpu().numpy(), labels=range(0, self.args.num_classes)
                    )

                    if self.confusion_matrix.shape[0] == 0:
                        self.confusion_matrix = conf
                    else:
                        self.confusion_matrix = self.confusion_matrix + conf

        return accuracy_meter_top_all.avg, accuracy_meter_top_body.avg, accuracy_meter_top_face.avg, loss_meter.avg, p * 100, r * 100, f * 100
