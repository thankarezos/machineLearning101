# -*- coding: utf-8 -*-
import mlp as mlp
import createDataset as cd

def main():

    n = None
    while True:
        try:
            n = int(input("Εισάγετε τον αριθμό των σημείων (πρέπει να είναι πολλαπλάσιο του 8): "))
            if n != "" and int(n) % 8 != 0:
                print("Ο αριθμός πρέπει να είναι πολλαπλάσιο του 8. Δοκιμάστε το", ((int(n)//8)+1)*8, "την επόμενη φορά.")
            else:
                break
        except ValueError:
            print("Invalid input! Please enter an integer.")

    print("1. Γραμμικά Διαχωρίσιμα Πρότυπα\n" +
    "2. Μη Γραμμικά Διαχωρίσιμα Πρότυπα – Κλάση 0 στη Γωνία\n" +
    "3. Μη Γραμμικά Διαχωρίσιμα Πρότυπα - Κλάση 0 στο Κέντρο\n" +
    "4. Μη Γραμμικά Διαχωρίσιμα Πρότυπα, Πύλη XOR\n" +
    "5. Μη Γραμμικά Διαχωρίσιμα Πρότυπα, Κλάση 0 μέσα στην Κλάση 1\n" +
    "6. Τέλος"
    )


    data = None
    while data not in range(1, 7):
        try:
            data = int(input("Επιλέξτε έναν αριθμό από 1 έως 6: "))
        except ValueError:
            print("Invalid input! Please enter an integer.")
    
    if data == 6:
        return
    num_epochs = 100

    while True:
        try:
            num_epochs = int(input(f"Enter the number of epochs (default={num_epochs}): ") or num_epochs)
            if num_epochs <= 0:
                print("The number of epochs must be a positive integer!")
            else:
                break
        except ValueError:
            print("Invalid input! Please enter an integer.")

    lr = 0.1

    while True:
        try:
            lr = float(input(f"Enter the learning rate (default={lr}): ") or lr)
            if lr <= 0:
                print("The learning rate must be a positive number!")
            else:
                break
        except ValueError:
            print("Invalid input! Please enter a number.")

    if data == 1:
        X = cd.linearSeparated(n)
    elif data == 2:
        X = cd.nonLinearAngle(n)
    elif data == 3:
        X = cd.nonLinearCenter(n)
    elif data == 4:
        X = cd.nonLinearXOR(n)
    elif data == 5:
        X = cd.nonLinear(n)
        
    print("1. GradientDecent\n" +
    "2. Stochastic Gradient Descent\n" +
    "3. LBFGS\n" +
    "4. Adam\n" +
    "5. Τέλος"
    )
    option = None
    while option not in range(1, 6):
        try:
            option = int(input("Επιλέξτε έναν αριθμό από 1 έως 6: "))
        except ValueError:
            print("Invalid input! Please enter an integer.")
    if option == 5:
        return
    
    if option == 1:
        print(data)
        if data == 1:
            hidden_layer_sizes=(1,)
            mlp.GD(X, hidden_layer_sizes=hidden_layer_sizes, max_iter=num_epochs, learning_rate_init=lr)
        elif data == 2 or data == 4:
            hidden_layer_sizes=(2,)
            mlp.GD(X, hidden_layer_sizes=hidden_layer_sizes, max_iter=num_epochs, learning_rate_init=lr)
        elif data == 3:
            hidden_layer_sizes=(4,)
            mlp.GD(X, hidden_layer_sizes=hidden_layer_sizes, max_iter=num_epochs, learning_rate_init=lr)
        elif data == 5:
            hidden_layer_sizes=(20,)
            mlp.GD(X, hidden_layer_sizes=hidden_layer_sizes, max_iter=num_epochs, learning_rate_init=lr)
        else:
            mlp.GD(X, max_iter=num_epochs, learning_rate_init=lr)
    elif option == 2:
        mlp.SGD(X, max_iter=num_epochs, learning_rate_init=lr)
    elif option == 3:
        mlp.LBFGS(X, max_iter=num_epochs, learning_rate_init=lr)
    elif option == 4:
        mlp.Adam(X, max_iter=num_epochs, learning_rate_init=lr)



if __name__ == "__main__":
    print("Εργασία 1 - Μηχανική Μάθηση\n")
    main()