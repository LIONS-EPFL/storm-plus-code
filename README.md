# storm-plus-code
PyTorch optimizer implementation for STORM+ algorithm

# How to use the optimizer?
With respect to mainstream PyTorch optimizers like SGD, our STORM+ requires two backward passes to compute two gradients using the same randomness/data sample at two different point/model weights. To do so, STORM+ optimizer has an auxiliary function called `compute_estimator()` that takes care of this extra task. We will give a brief description of how to loop over optimizer steps:

```python
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
    optimizer.compute_estimator()
```
