import torch


def train_model(train_loader, test_loader, model, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_loss_list = []
    training_accuracy_list = []
    test_accuracy_list = []
    epochs_list = []
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels-1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            predicted = predicted+1
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            print(f'Current Epoch Train Accuracy: {(correct_train/total_train)*100:.2f}%')

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # Evaluation phase
        model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted+1
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                print(f'Current Epoch Test Accuracy: {(correct_test/total_test)*100:.2f}%')

        test_accuracy = 100 * correct_test / total_test

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Test Accuracy: {test_accuracy:.2f}%")

        training_loss_list.append(train_loss)
        training_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)
        epochs_list.append(epoch)

    return training_loss_list,training_accuracy_list,test_accuracy_list,epochs_list