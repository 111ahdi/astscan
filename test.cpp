#include <iostream>

// This is a test file for our new analyzer.
int main() {
    int a = 10;
    int* p1 = new int;        // init_declarator 场景
    double* p2 = new double[20];
    int* p3;
    p3 = new int;             // assignment_expression 场景

    delete p1;

    return 0;
}