MAT_SSVM
========

A MATLAB implementation of the structural SVM with the cutting-plane algorithm.

It solves 1-slack (or n-slack) structural SVM with margin-rescaling. 
The implementation tries to stay as close as possible to the interface of svm^struct Matlab. 
The resulted QP problem is simply solved via pegasos algorithm, and I don't use the sparse structure for feature map and weight vector.

Note that this is designed only for better understanding the principles of structural SVM, and it may be slower than the original implementation with C code and more efficient QP solver.

References:  
[1] T. Joachims, T. Finley, Chun-Nam Yu, Cutting-Plane Training of Structural SVMs, Machine Learning Journal.


Created by Chaoran Cui (bruincui@gmail.com)  
homepage: http://ir.sdu.edu.cn/~chaorancui/  
If there are any problems, please let me know.

