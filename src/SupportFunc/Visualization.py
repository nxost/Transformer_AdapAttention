import matplotlib.pyplot as plt
import seaborn as sns 

def plot_confusion_matrix(conf_matrix, class_name):
    #graficar la matriz de confusion
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Matriz de Confusión para la clase: {class_name}')
    plt.show()
    
def plot_train_val_curve(num_epochs, train_losses, val_losses):
    # Graficar curvas de entrenamiento y validación
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs+1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()