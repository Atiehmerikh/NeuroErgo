# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import poly_coeff_calculator as pCoeff

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    m_pcoeff = pCoeff.polynomial_generator()
    m_pcoeff.neck_polynomial([0,10,20,80],[0,2,1,2])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
