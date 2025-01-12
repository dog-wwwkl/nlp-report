import re
import matplotlib.pyplot as plt

# Read the log file
log_file_path = './llama-2-7b-0.75to0.25/stderr.log'  
log_file_path2 = './llama-2-7b-0.25to0.75/stderr.log'  
loss_values = []
iterations = []
loss_values2 = []
with open(log_file_path, 'r') as file:
    for line in file:
        # Use regular expressions to match iteration counts and loss values
        match = re.search(r'Training \d+/\d+ epoch \(loss ([-\d.]+)\):\s+\d+%.*?\| (\d+)/\d+ \[', line)
        if match:
            # Extract and convert loss values and iteration counts
            loss_value = float(match.group(1))
            iteration = int(match.group(2))
            iterations.append(iteration)
            loss_values.append(loss_value)
with open(log_file_path2, 'r') as file:
    for line in file:
        # Use regular expressions to match iteration counts and loss values
        match = re.search(r'Training \d+/\d+ epoch \(loss ([-\d.]+)\):\s+\d+%.*?\| (\d+)/\d+ \[', line)
        if match:
            # Extract and convert loss values
            loss_value = float(match.group(1))
            loss_values2.append(loss_value)

average1 = sum(loss_values) / len(loss_values)
average2 = sum(loss_values2) / len(loss_values2)

# Display average loss values
print('Average loss 0.75to0.25_1e-5_1epoch:', average1)
print('Average loss 0.25to0.75_1e-5_1epoch:', average2)

# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(iterations, loss_values, label='Training loss 0.75to0.25_1e-5_1epoch', color='blue')
plt.plot(iterations, loss_values2, label='Training loss 0.25to0.75_1e-5_1epoch', color='red')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)

# Save the image to a file
plt.savefig('loss_0.25and0.75_new_1e-5_1epoch.png')
plt.close()
