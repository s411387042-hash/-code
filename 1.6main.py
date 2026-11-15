# 1.6 Results output and Accuracy plot
print("\n--- 5x4 Accuracy Table ---")
print(f"{'Method':<10} | {'sa=1':<8} | {'sa=3':<8} | {'sa=10':<8}")
print("-" * 40)
for method, accs in results.items():
    print(f"{method:<10} | {accs[0]:.4f} | {accs[1]:.4f} | {accs[2]:.4f}")

plt.figure(figsize=(8, 5))
for method, accs in results.items():
    plt.plot(sigma_a_list, accs, marker='o', label=method)
plt.title('Test Accuracy vs sigma_a')
plt.xlabel('sigma_a')
plt.ylabel('Accuracy')
plt.xticks(sigma_a_list)
plt.legend()
plt.grid(True)

# 改成存檔：
plt.savefig('1_6_Accuracy_Plot.png')
plt.close()
