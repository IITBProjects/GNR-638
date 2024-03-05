import torch


def train_model(train_loader, test_loader, model, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_loss_list = []
    training_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []
    epochs_list = []
    for epoch in range(num_epochs):
        print(f"Starting Epoch [{epoch+1}/{num_epochs}]")
        # Training phase
        model.train()
        running_loss = 0.0
        train_total_count = 0
        train_total_correct = 0
        batch_number = 0
        total_batches = len(train_loader)

        for inputs, labels in train_loader:
            batch_number += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels-1)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            predicted = predicted+1
            train_batch_count = labels.size(0)
            train_batch_correct = (predicted == labels).sum().item()
            train_total_count += train_batch_count
            train_total_correct += train_batch_correct
            running_loss += train_batch_count*loss.item()

            print(f'Train Batch [{batch_number}/{total_batches}] :      Batch Size : {train_batch_count}        Batch Mean Loss: {loss.item():.5f}      Batch Accuracy: {(train_batch_correct/train_batch_count)*100:.2f}%     Rolling Epoch Accuracy: {(train_total_correct/train_total_count)*100:.2f}%')

        # Evaluation phase
        model.eval()
        
        with torch.no_grad():
            train_total_count = 0
            train_total_correct = 0
            total_train_loss = 0
            batch_number = 0
            total_batches = len(train_loader)
            for inputs, labels in train_loader:
                batch_number += 1
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels-1)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted+1
                train_batch_count = labels.size(0)
                train_batch_correct = (predicted == labels).sum().item()
                total_train_loss += loss.item()*train_batch_count
                train_total_count += train_batch_count
                train_total_correct += train_batch_correct
                print(f'Evaluation Train Batch [{batch_number}/{total_batches}] :      Batch Size : {train_batch_count}     Batch Mean Loss: {loss.item():.5f}     Batch Accuracy: {(train_batch_correct/train_batch_count)*100:.2f}%')

            test_total_count = 0
            test_total_correct = 0
            batch_number = 0
            total_batches = len(test_loader)
            total_test_loss = 0
            for inputs, labels in test_loader:
                batch_number += 1
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels-1)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted+1
                test_batch_count = labels.size(0)
                test_batch_correct = (predicted == labels).sum().item()
                total_test_loss += loss.item()*test_batch_count
                test_total_count += test_batch_count
                test_total_correct += test_batch_correct
                print(f'Evaluation Test Batch [{batch_number}/{total_batches}] :      Batch Size : {test_batch_count}     Batch Mean Loss: {loss.item():.5f}     Batch Accuracy: {(test_batch_correct/test_batch_count)*100:.2f}%')
            
            train_accuracy = 100 * train_total_correct / train_total_count
            test_accuracy = 100 * test_total_correct / test_total_count


        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Mean Train Loss: {total_train_loss/train_total_count:.4f}, Mean Test Loss: {total_test_loss/test_total_count:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%, "
              f"Train Size: {train_total_count}, Test Size: {test_total_count}")

        training_loss_list.append(total_train_loss/train_total_count)
        training_accuracy_list.append(train_accuracy)
        test_loss_list.append(total_test_loss/test_total_count)
        test_accuracy_list.append(test_accuracy)
        epochs_list.append(epoch)

    return training_loss_list,training_accuracy_list,test_loss_list,test_accuracy_list,epochs_list