import torch
import os

def test(model, test_loader, criterion, device):
    with torch.no_grad():
        true_num = 0
        test_loss = 0
        for test_data, test_label in test_loader:
            test_data = test_data.to(device)
            test_label = test_label.to(device)
            test_out = model(test_data)
            loss = criterion(test_out, test_label)
            test_loss += loss.item()
            # print(test_out.shape,test_label.shape) #[64,10],[64]
            num = (torch.argmax(test_out, dim=1) == test_label).sum().item()
            true_num += num
        print('Test Loss: {:.6f}, Test Accuracy: {:.3f}%'.format(test_loss / len(test_loader), 100 * true_num / len(test_loader.dataset)))

def train(model, train_loader, num_epochs, optimizer, criterion, device, test_loader=None, saved_model_path=None):
    for epoch in range(num_epochs):
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # 将图像数据展平为一维向量
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()

             # 反向传播并更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印训练信息
            if (i + 1) % 100 == 0:
                print('Epoch:{} | Step:{}/{} | Loss: {:.6f}'.format(epoch + 1, i + 1, \
                                        len(train_loader), train_loss / 100))
                train_loss = 0

        # test
        if test_loader:
            test(model, test_loader, criterion, device)
        if saved_model_path:
            saved_epoch_model_path = os.path.join(saved_model_path, f"model{epoch + 1}.pth")
            torch.save(model, saved_epoch_model_path)
