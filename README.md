# storm-plus-code
PyTorch optimizer implementation for STORM+ algorithm

# How to use the optimizer?
On the contrary to mainstream (PyTorch) optimizers like SGD, our STORM+ (and STORM algorith itself) requires two backward passes to compute two gradients using the same randomness/data sample at two different point/model weights. To do so, STORM+ optimizer has an auxiliary function called `compute_estimator()` that takes care of this extra task. We will give a brief description of how to loop over optimizer steps.

```python
for epoch in range(num_epochs):

    # Generate data iterator, CIFAR10 in this case
    train_data_loader = torch.utils.data.DataLoader(CIFAR10_train_set, ...)

    # Enter train mode
    model.train()

    # ONLY FOR THE FIRST BATCH: We need to compute the initial estimator d_1, 
    # which is the first (mini-batch) stochastic gradient g_1. To set the estimator
    # we need to call compute_step() with the first batch.
    (data, label) = iter(train_data_loader).next()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, label)
    loss.backward()
    optimizer.compute_estimator(normalized_norm=True)

    for i, (data, label) in enumerate(training_data, 0):


        # main optimization step
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()

        # uses \tilde g_t from the backward() call above
        # uses d_t already saved as parameter group state from previous iteration
        optimizer.step()  

        # makes the second pass, backpropagation for the NEXT iterate using the current data batch
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()

        # updates estimate d_{t+1} for the next iteration, saves g_{t+1} for next iteration
        optimizer.compute_estimator(normalized_norm=True)
```
