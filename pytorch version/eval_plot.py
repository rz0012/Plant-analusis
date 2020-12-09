import pandas as pd
import matplotlib.pyplot as plt
fname = input("please input generated file location: ")
data = pd.read_csv(fname)

print()
print(data)
print()

plt.figure(figsize=(10, 10))
plt.plot(data["Epoch"],data["Train Accuracy"], label='Train Accuracy')
plt.plot(data["Epoch"],data["Test Accuracy"], label='Test Accuracy')
plt.legend()
plt.show()


plt.figure(figsize=(10, 10))
plt.plot(data["Epoch"],data["Train Loss"], label='Train Loss')
plt.plot(data["Epoch"],data["Test Loss"], label='Test Loss')
plt.legend()
plt.show()

