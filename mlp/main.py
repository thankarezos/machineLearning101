# -*- coding: utf-8 -*-
import mlp as mlp

def main():

    
    # lr = model.learning_rate

    # n = None
    # while True:
    #     try:
    #         n = int(input("Εισάγετε τον αριθμό των σημείων (πρέπει να είναι πολλαπλάσιο του 8): "))
    #         if n != "" and int(n) % 8 != 0:
    #             print("Ο αριθμός πρέπει να είναι πολλαπλάσιο του 8. Δοκιμάστε το", ((int(n)//8)+1)*8, "την επόμενη φορά.")
    #         else:
    #             break
    #     except ValueError:
    #         print("Invalid input! Please enter an integer.")

    print("1. Γραμμικά Διαχωρίσιμα Πρότυπα\n" +
    "2. Μη Γραμμικά Διαχωρίσιμα Πρότυπα – Κλάση 0 στη Γωνία\n" +
    "3. Μη Γραμμικά Διαχωρίσιμα Πρότυπα - Κλάση 0 στο Κέντρο\n" +
    "4. Μη Γραμμικά Διαχωρίσιμα Πρότυπα, Πύλη XOR\n" +
    "5. Μη Γραμμικά Διαχωρίσιμα Πρότυπα, Κλάση 0 μέσα στην Κλάση 1\n" +
    "6. Τέλος"
    )


    option = None
    while option not in range(1, 7):
        try:
            option = int(input("Επιλέξτε έναν αριθμό από 1 έως 6: "))
        except ValueError:
            print("Invalid input! Please enter an integer.")
    
    if option == 6:
        return

    while True:
        try:
            num_epochs = int(input("Enter the number of epochs (default=100): ") or 100)
            if num_epochs <= 0:
                print("The number of epochs must be a positive integer!")
            else:
                break
        except ValueError:
            print("Invalid input! Please enter an integer.")

    while True:
        try:
            lr = float(input(f"Enter the learning rate (default={lr}): ") or lr)
            if lr <= 0:
                print("The learning rate must be a positive number!")
            else:
                break
        except ValueError:
            print("Invalid input! Please enter a number.")

    if option == 1:
        tr.linearSeparated(n, model)
    elif option == 2:
        tr.nonlinearSeparatedAngle(n, model)
    elif option == 3:
        tr.nonlinearSeparatedCenter(n, model)
    elif option == 4:
        tr.nonlinearSeparatedXOR(n, model)
    elif option == 5:
        tr.nonlinearSeparated(n, model)
        
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


if __name__ == "__main__":
    print("Εργασία 1 - Μηχανική Μάθηση\n")
    main()