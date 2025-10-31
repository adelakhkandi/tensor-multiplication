Tensor Multiplication Examples

A compact, runnable introduction to tensor multiplication with NumPy.
The notebook demonstrates six core operations—element-wise, dot, matrix, outer, Kronecker, and tensor contraction—with clear explanations and code.

📂 Contents

tensor_multiplication_examples.ipynb — Main Jupyter notebook with explanations and executable cells.

🔧 Requirements

Python 3.8+

NumPy

(Optional) Jupyter / VS Code / Google Colab

Install locally:

pip install numpy jupyter

▶️ Quick Start

Open the notebook locally:

jupyter notebook tensor_multiplication_examples.ipynb


Or upload the .ipynb to Google Colab and run cell by cell.

🧮 What’s Inside

Element-wise Multiplication
Multiplies corresponding elements (same-shaped tensors).

C = A * B


Dot Product (Vectors)
Sum of element-wise products → scalar.

np.dot(a, b)


Matrix Multiplication
Standard row×column product.

A @ B


Outer Product
Every element of a multiplies every element of b → matrix.

np.outer(a, b)


Kronecker Product
Expanded outer product for matrices → larger matrix.

np.kron(A, B)


Tensor Contraction
Generalization of matrix multiplication; sums over shared axes.

np.tensordot(A, B, axes=([2],[0]))

📘 Example Snippet
import numpy as np

A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

print("Element-wise:\n", A * B)
print("Matrix product:\n", A @ B)
print("Kronecker:\n", np.kron(A, B))

🎯 Purpose

Provide a concise, hands-on reference for common tensor multiplications.

Serve as a starting point for students and practitioners working with linear algebra, machine learning, and deep learning.

📄 License

MIT — feel free to use and modify.
