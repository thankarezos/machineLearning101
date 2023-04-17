# -*- coding: utf-8 -*-

import train as tr

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
    
    # while True:
    #     try:
    #         n = int(input("Εισάγετε τον αριθμό των σημείων (πρέπει να είναι πολλαπλάσιο του 8): "))
    #         if n != "" and int(n) % 8 != 0:
    #             print("Ο αριθμός πρέπει να είναι πολλαπλάσιο του 8. Δοκιμάστε το", ((int(n)//8)+1)*8, "την επόμενη φορά.")
    #         else:
    #             break
    #     except ValueError:
    #         print("Invalid input! Please enter an integer.")


    option = None
    while option not in range(1, 7):
        try:
            option = int(input("Επιλέξτε έναν αριθμό από 1 έως 6: "))
        except ValueError:
            print("Invalid input! Please enter an integer.")

    if option == 1:
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
                lr = float(input("Enter the learning rate (default=0.1): ") or 0.1)
                if lr <= 0:
                    print("The learning rate must be a positive number!")
                else:
                    break
            except ValueError:
                print("Invalid input! Please enter a number.")

        tr.linearSeperated(n, num_epochs=num_epochs, learning_rate=lr)



if __name__ == "__main__":
    main()