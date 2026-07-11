#include "Tensor.h"
void printGrad(Tensor &t){
    for (int i = 0; i < t.rows(); ++i) {
        for (int j = 0; j < t.cols(); ++j)
            std::cout << t.grad()[i * t.cols() + j] << " ";
        std::cout << "\n";
    }
}

int main(){
    Tensor a(2, 2, true);
    Tensor b(2, 2, true);

    a.at(0,0)=1; b.at(0,0)=2;
    a.at(0,1)=3; b.at(0,1)=4;

//加个打印功能
    Tensor c = add(a, b);
    c.backward();

    std::cout << a.grad()[0] << " " << a.grad()[1] << "\n";


    Tensor A(2, 3, true);
    Tensor B(3, 2, true);

    A.at(0,0)=1; A.at(0,1)=2; A.at(0,2)=3;
    A.at(1,0)=4; A.at(1,1)=5; A.at(1,2)=6;

    B.at(0,0)=7; B.at(0,1)=8;
    B.at(1,0)=9; B.at(1,1)=10;
    B.at(2,0)=11; B.at(2,1)=12;
    Tensor C = matmul(A, B);
    C.backward();
    std::cout << "Gradient of A:\n";
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j)
            std::cout << A.grad()[i * A.cols() + j] << " ";
        std::cout << "\n";
    }

    Tensor subA(2,2,true);
    Tensor subB(2,2,true);
    subA.at(0,0)=10; subA.at(0,1)=4;
    subA.at(1,0)=3;  subA.at(1,1)=2;

    subB.at(0,0)=2; subB.at(0,1)=2;
    subB.at(1,0)=4;  subB.at(1,1)=8;

    Tensor subr=sub(subA,subB);
    subr.backward();

    std::cout << "Gradient of subA:\n";
    printGrad(subA);

    Tensor divA(2,2,true);
    Tensor divB(2,2,true);
    divA.at(0,0)=10; divA.at(0,1)=4;
    divA.at(1,0)=3;  divA.at(1,1)=2;

    divB.at(0,0)=2; divB.at(0,1)=2;
    divB.at(1,0)=4;  divB.at(1,1)=8;

    Tensor divr=div(divA,divB);
    divr.backward();
    std::cout << "Gradient of divA:\n";
    printGrad(divA);

    
}







