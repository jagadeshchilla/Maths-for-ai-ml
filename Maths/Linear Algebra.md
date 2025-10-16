
# Linear Algebra Concepts (Comprehensive Guide)

This document provides a comprehensive overview of linear algebra concepts with detailed explanations, mathematical formulations, examples, and applications.

---

## 1️⃣ Vectors in Linear Algebra

A **vector** is an ordered collection of numbers representing a point, direction, or quantity in space.

### Definition
A vector **v** in ℝⁿ is an ordered n-tuple:

```
v = [v₁, v₂, ..., vₙ]ᵀ = (v₁, v₂, ..., vₙ)
```

**Column vector notation:**
```
v = [v₁]
    [v₂]
    [⋮]
    [vₙ]
```

### Basic Properties
- **Notation:** **v** = [v₁, v₂, ..., vₙ] or **v** = (v₁, v₂, ..., vₙ)
- **Dimension:** number of components (2D, 3D, n-dimensional)
- **Geometric meaning:** points to a direction and has magnitude
- **Zero vector:** **0** = [0, 0, ..., 0]ᵀ

### Vector Operations

#### Addition
For vectors **v** = [v₁, v₂, ..., vₙ]ᵀ and **w** = [w₁, w₂, ..., wₙ]ᵀ:

```
v + w = [v₁ + w₁]
        [v₂ + w₂]
        [  ⋮   ]
        [vₙ + wₙ]
```

**Properties:**
- **Commutative:** v + w = w + v
- **Associative:** (u + v) + w = u + (v + w)
- **Identity:** v + 0 = v

#### Scalar Multiplication
For scalar c and vector **v**:

```
cv = [cv₁]
     [cv₂]
     [ ⋮ ]
     [cvₙ]
```

**Properties:**
- **Associative:** (ab)v = a(bv)
- **Distributive:** a(v + w) = av + aw
- **Distributive:** (a + b)v = av + bv

#### Dot Product (Inner Product)
For vectors **v** and **w**:

```
v · w = v₁w₁ + v₂w₂ + ... + vₙwₙ = Σᵢ₌₁ⁿ vᵢwᵢ
```

**Geometric interpretation:**
```
v · w = |v||w|cos(θ)
```
where θ is the angle between vectors.

**Properties:**
- **Commutative:** v · w = w · v
- **Distributive:** u · (v + w) = u · v + u · w
- **Scalar multiplication:** (cv) · w = c(v · w)

#### Vector Magnitude (Norm)
The **Euclidean norm** (or magnitude) of vector **v**:

```
|v| = √(v · v) = √(v₁² + v₂² + ... + vₙ²)
```

#### Unit Vector
A **unit vector** in the direction of **v**:

```
v̂ = v/|v|
```

**Properties:**
- |v̂| = 1
- v̂ points in the same direction as v

### Applications
- **Physics:** Force, velocity, acceleration vectors
- **Computer Graphics:** 3D transformations, lighting calculations
- **Machine Learning:** Feature vectors, similarity measures

---

## 2️⃣ Linear Combinations, Span, and Basis

### Linear Combinations

A **linear combination** of vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ is any vector of the form:
$$\mathbf{w} = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k$$
where $c_1, c_2, \ldots, c_k$ are scalars.

**Example:**
If $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ and $\mathbf{v}_2 = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$, then:
$$2\mathbf{v}_1 + 3\mathbf{v}_2 = 2\begin{bmatrix} 1 \\ 2 \end{bmatrix} + 3\begin{bmatrix} 3 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 4 \end{bmatrix} + \begin{bmatrix} 9 \\ 3 \end{bmatrix} = \begin{bmatrix} 11 \\ 7 \end{bmatrix}$$

### Linear Independence

Vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ are **linearly independent** if:
$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0}$$
only when $c_1 = c_2 = \cdots = c_k = 0$.

**Test for linear independence:**
- For 2D vectors: $\mathbf{v}_1$ and $\mathbf{v}_2$ are independent if $\mathbf{v}_1 \neq k\mathbf{v}_2$ for any scalar $k$
- For 3D vectors: $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$ are independent if $\det([\mathbf{v}_1 \mathbf{v}_2 \mathbf{v}_3]) \neq 0$

### Span

The **span** of vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ is the set of all possible linear combinations:
$$\text{Span}\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\} = \{c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k : c_i \in \mathbb{R}\}$$

**Geometric interpretation:** 
- Span of one vector: a line through the origin
- Span of two independent vectors: a plane through the origin
- Span of three independent vectors in $\mathbb{R}^3$: all of $\mathbb{R}^3$

### Basis

A **basis** for a vector space $V$ is a set of vectors that:
1. **Spans V:** Every vector in $V$ can be written as a linear combination of basis vectors
2. **Is linearly independent:** No basis vector can be written as a linear combination of the others

**Standard basis for $\mathbb{R}^n$:**
$$\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, \quad \mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, \quad \ldots, \quad \mathbf{e}_n = \begin{bmatrix} 0 \\ 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}$$

### Dimension

The **dimension** of a vector space is the number of vectors in any basis for that space.

**Properties:**
- $\dim(\mathbb{R}^n) = n$
- $\dim(\text{Span}\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}) =$ number of linearly independent vectors in $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$

### Examples

**Example 1:** In $\mathbb{R}^2$
- $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$, $\mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$ form a basis
- Any vector $\begin{bmatrix} a \\ b \end{bmatrix} = a\begin{bmatrix} 1 \\ 0 \end{bmatrix} + b\begin{bmatrix} 0 \\ 1 \end{bmatrix}$

**Example 2:** In $\mathbb{R}^3$
- $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}$, $\mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}$ span a plane
- Adding $\mathbf{v}_3 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}$ gives a basis for $\mathbb{R}^3$

### Applications
- **Computer Graphics:** Coordinate systems, transformations
- **Machine Learning:** Feature spaces, dimensionality reduction
- **Physics:** Coordinate transformations, quantum mechanics

---

## 3️⃣ Linear Transformations and Matrices

### Linear Transformations

A **linear transformation** $T: \mathbb{R}^n \to \mathbb{R}^m$ is a function that maps vectors from one space to another while preserving vector addition and scalar multiplication.

**Definition:**
$$T(a\mathbf{u} + b\mathbf{v}) = aT(\mathbf{u}) + bT(\mathbf{v})$$
for all vectors $\mathbf{u}, \mathbf{v}$ and scalars $a, b$.

**Properties:**
- $T(\mathbf{0}) = \mathbf{0}$ (maps zero vector to zero vector)
- $T(-\mathbf{v}) = -T(\mathbf{v})$ (preserves negation)

### Matrix Representation

Every linear transformation $T: \mathbb{R}^n \to \mathbb{R}^m$ can be represented by an $m \times n$ matrix $A$:
$$T(\mathbf{x}) = A\mathbf{x}$$

**How to find the matrix:**
1. Apply $T$ to each standard basis vector $\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n$
2. The columns of $A$ are $T(\mathbf{e}_1), T(\mathbf{e}_2), \ldots, T(\mathbf{e}_n)$

### Types of Matrices

#### Square Matrices ($n \times n$)
- **Identity matrix:** $I = \text{diag}(1, 1, \ldots, 1)$
- **Diagonal matrix:** $D = \text{diag}(d_1, d_2, \ldots, d_n)$
- **Symmetric matrix:** $A = A^T$
- **Skew-symmetric matrix:** $A = -A^T$
- **Orthogonal matrix:** $A^TA = I$ (preserves lengths and angles)

#### Special Matrices
- **Zero matrix:** $O$ (all entries are 0)
- **Upper triangular:** entries below main diagonal are 0
- **Lower triangular:** entries above main diagonal are 0
- **Permutation matrix:** exactly one 1 in each row/column

### Matrix Operations

#### Addition
For matrices $A$ and $B$ of the same size:
$$(A + B)_{ij} = A_{ij} + B_{ij}$$

**Properties:**
- **Commutative:** $A + B = B + A$
- **Associative:** $(A + B) + C = A + (B + C)$

#### Scalar Multiplication
For scalar $c$ and matrix $A$:
$$(cA)_{ij} = c \cdot A_{ij}$$

#### Matrix Multiplication
For $A$ ($m \times n$) and $B$ ($n \times p$):
$$(AB)_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}$$

**Properties:**
- **Associative:** $(AB)C = A(BC)$
- **Distributive:** $A(B + C) = AB + AC$
- **NOT commutative:** $AB \neq BA$ in general

#### Transpose
$$(A^T)_{ij} = A_{ji}$$

**Properties:**
- $(A^T)^T = A$
- $(A + B)^T = A^T + B^T$
- $(AB)^T = B^TA^T$

### Examples

**Example 1: Rotation Matrix (2D)**
$$R(\theta) = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}$$

**Example 2: Scaling Matrix**
$$S = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$

**Example 3: Reflection Matrix (across x-axis)**
$$F = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$$

### Applications
- **Computer Graphics:** 3D transformations, rotations, scaling
- **Physics:** Coordinate transformations, quantum mechanics
- **Engineering:** Structural analysis, signal processing
- **Machine Learning:** Feature transformations, dimensionality reduction

---

## 4️⃣ Matrix Multiplication as Composition

Matrix multiplication represents **composition of linear transformations**.

### Composition of Transformations

If T₁: ℝⁿ → ℝᵐ and T₂: ℝᵐ → ℝᵖ are linear transformations with matrices A and B respectively, then:
```
T₂(T₁(x)) = T₂(Ax) = B(Ax) = (BA)x
```

**Key insight:** The matrix BA represents doing T₁ first, then T₂.

### Order Matters

**Important:** Matrix multiplication is not commutative!
- AB represents: "do B first, then A"
- BA represents: "do A first, then B"

### Examples

**Example 1: Rotation followed by Scaling**
```
R = [cos(θ) -sin(θ)]  (rotation)
    [sin(θ)  cos(θ)]

S = [2  0]            (scaling)
    [0  3]

SR = [2cos(θ) -2sin(θ)]  (scale then rotate)
     [3sin(θ)  3cos(θ)]

RS = [2cos(θ) -3sin(θ)]  (rotate then scale)
     [2sin(θ)  3cos(θ)]
```

**Example 2: Multiple Transformations**
For transformations T₁, T₂, T₃ with matrices A₁, A₂, A₃:
```
T₃(T₂(T₁(x))) = A₃(A₂(A₁x)) = (A₃A₂A₁)x
```

### Properties of Composition

1. **Associative:** (AB)C = A(BC)
2. **Identity:** AI = IA = A
3. **Distributive:** A(B + C) = AB + AC

### Applications
- **Computer Graphics:** Combining multiple transformations
- **Robotics:** Kinematic chains, coordinate transformations
- **Physics:** Successive coordinate system changes

---

## 5️⃣ Three-Dimensional Linear Transformations

In 3D space, linear transformations can be represented by 3×3 matrices acting on vectors [x, y, z]ᵀ.

### Common 3D Transformations

#### Rotation Matrices

**Rotation around x-axis:**
```
Rₓ(θ) = [1     0        0   ]
        [0  cos(θ) -sin(θ)]
        [0  sin(θ)  cos(θ)]
```

**Rotation around y-axis:**
```
Rᵧ(θ) = [ cos(θ)  0  sin(θ)]
        [   0     1    0   ]
        [-sin(θ)  0  cos(θ)]
```

**Rotation around z-axis:**
```
Rᵧ(θ) = [cos(θ) -sin(θ)  0]
        [sin(θ)  cos(θ)  0]
        [  0       0     1]
```

#### Scaling Matrix
```
S = [sₓ  0   0 ]
    [ 0  sᵧ  0 ]
    [ 0   0  sᵧ]
```

#### Reflection Matrices

**Reflection across xy-plane:**
```
Fₓᵧ = [1  0  0]
      [0  1  0]
      [0  0 -1]
```

#### Shear Matrices

**Shear in x-direction:**
```
Hₓ = [1  hₓᵧ  hₓᵧ]
     [0   1    0 ]
     [0   0    1 ]
```

### Composition of 3D Transformations

**Example: Rotation around arbitrary axis**
To rotate around axis u = [uₓ, uᵧ, uᵧ] by angle θ:
```
R = cos(θ)I + sin(θ)[u]× + (1-cos(θ))(u ⊗ u)
```
where [u]× is the skew-symmetric matrix and u ⊗ u is the outer product.

### Applications
- **Computer Graphics:** 3D modeling, animation, rendering
- **Robotics:** Robot arm kinematics, coordinate transformations
- **Physics:** Rigid body dynamics, celestial mechanics
- **Engineering:** Structural analysis, CAD systems

---

## 6️⃣ The Determinant

The **determinant** is a scalar value that provides important information about a square matrix and its associated linear transformation.

### Definition
For an $n \times n$ matrix $A$, the determinant is defined recursively:
$$\det(A) = \sum_{j=1}^{n} (-1)^{i+j} a_{ij} \det(A_{ij})$$
where $A_{ij}$ is the $(n-1) \times (n-1)$ matrix obtained by deleting row $i$ and column $j$.

### Geometric Interpretation

The determinant measures how a linear transformation scales area (2D) or volume (3D):
- **2D:** $\det(A) =$ area scaling factor
- **3D:** $\det(A) =$ volume scaling factor
- **nD:** $\det(A) =$ $n$-dimensional volume scaling factor

### Properties of Determinants

1. **Zero determinant:** $\det(A) = 0 \iff A$ is singular (not invertible)
2. **Sign:** 
   - $\det(A) > 0 \iff$ orientation preserved
   - $\det(A) < 0 \iff$ orientation reversed
3. **Multiplicative:** $\det(AB) = \det(A)\det(B)$
4. **Transpose:** $\det(A^T) = \det(A)$
5. **Scalar multiplication:** $\det(cA) = c^n\det(A)$ for $n \times n$ matrix $A$

### Computing Determinants

#### 2×2 Matrix
$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

#### 3×3 Matrix (Sarrus' Rule)
$$\det\begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix} = a(ei - fh) - b(di - fg) + c(dh - eg)$$

#### General n×n Matrix (Laplace Expansion)
$$\det(A) = \sum_{j=1}^{n} (-1)^{i+j} a_{ij} \det(A_{ij})$$
where $A_{ij}$ is the matrix obtained by deleting row $i$ and column $j$.

### Cofactor Expansion

For any row $i$:
$$\det(A) = \sum_{j=1}^{n} a_{ij} C_{ij}$$
where $C_{ij} = (-1)^{i+j} \det(A_{ij})$ is the cofactor.

### Examples

**Example 1: 2×2 Matrix**
$$A = \begin{pmatrix} 3 & 1 \\ 2 & 4 \end{pmatrix}$$
$$\det(A) = 3 \cdot 4 - 1 \cdot 2 = 12 - 2 = 10$$

**Example 2: 3×3 Matrix**
$$A = \begin{pmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 0 & 0 & 6 \end{pmatrix}$$
$$\det(A) = 1 \cdot 4 \cdot 6 = 24 \quad \text{(triangular matrix)}$$

### Applications

1. **Invertibility:** A is invertible ⟺ det(A) ≠ 0
2. **Volume/Area:** Computing volumes of parallelepipeds
3. **Change of variables:** Jacobian determinant in calculus
4. **Eigenvalues:** Characteristic polynomial roots
5. **Cross product:** |v × w| = |v||w|sin(θ) = det([v w])

---

## 7️⃣ Inverse Matrices, Column Space, and Null Space

### Inverse Matrices

The **inverse** of a square matrix A is a matrix A⁻¹ such that:
```
A·A⁻¹ = A⁻¹·A = I
```

#### Conditions for Invertibility
A matrix A is invertible if and only if:
- det(A) ≠ 0
- A has full rank
- The columns of A are linearly independent
- A is nonsingular

#### Computing the Inverse

**For 2×2 matrices:**
```
A⁻¹ = (1/det(A)) [ d -b]
                 [-c  a]
```

**For general matrices (Gauss-Jordan elimination):**
1. Form the augmented matrix [A | I]
2. Apply row operations to get [I | A⁻¹]

#### Properties of Inverses
- (A⁻¹)⁻¹ = A
- (AB)⁻¹ = B⁻¹A⁻¹
- (Aᵀ)⁻¹ = (A⁻¹)ᵀ
- det(A⁻¹) = 1/det(A)

### Column Space (Range)

The **column space** Col(A) of matrix A is the span of its columns:
```
Col(A) = Span{a₁, a₂, ..., aₙ}
```
where aᵢ is the i-th column of A.

**Properties:**
- Col(A) = {Ax : x ∈ ℝⁿ} (all possible outputs)
- dim(Col(A)) = rank(A)
- Col(A) ⊆ ℝᵐ (where A is m×n)

### Null Space (Kernel)

The **null space** Null(A) of matrix A is:
```
Null(A) = {x ∈ ℝⁿ : Ax = 0}
```

**Properties:**
- Null(A) is a subspace of ℝⁿ
- dim(Null(A)) = n - rank(A)
- Null(A) = {0} ⟺ A has full column rank

### Rank-Nullity Theorem

For an m×n matrix A:
```
rank(A) + nullity(A) = n
```
where nullity(A) = dim(Null(A)).

### Examples

**Example 1: Finding Column Space**
```
A = [1 2]
    [3 4]
Col(A) = Span{[1,3], [2,4]} = ℝ²
```

**Example 2: Finding Null Space**
```
A = [1 2 3]
    [0 1 2]
Null(A) = {x : x₁ + 2x₂ + 3x₃ = 0, x₂ + 2x₃ = 0}
        = Span{[1, -2, 1]}
```

### Applications
- **Solving linear systems:** Ax = b has solution ⟺ b ∈ Col(A)
- **Linear independence:** Vectors are independent ⟺ null space is {0}
- **Dimensionality:** Understanding the "size" of solution spaces
- **Machine Learning:** Feature spaces, principal component analysis

---

## 8️⃣ Nonsquare Matrices as Transformations Between Dimensions

An m×n matrix A represents a linear transformation from ℝⁿ to ℝᵐ.

### Dimension Changes

#### Embedding (m > n)
When m > n, the transformation **embeds** a lower-dimensional space into a higher-dimensional one.

**Example:**
```
A = [1 0]    (2×1 matrix)
    [0 1]
    [1 1]
```
Maps ℝ¹ → ℝ³: x ↦ [x, 0, x]ᵀ

#### Projection (m < n)
When m < n, the transformation **projects** a higher-dimensional space onto a lower-dimensional one.

**Example:**
```
A = [1 0 0]  (1×3 matrix)
```
Maps ℝ³ → ℝ¹: [x, y, z]ᵀ ↦ x

### Rank and Dimension

For an m×n matrix A:
- **Column rank:** dim(Col(A)) ≤ min(m, n)
- **Row rank:** dim(Row(A)) ≤ min(m, n)
- **Rank:** rank(A) = column rank = row rank

### Examples

**Example 1: 3×2 Matrix (Embedding)**
```
A = [1 0]
    [0 1]
    [1 1]
```
- Maps ℝ² → ℝ³
- Col(A) = plane in ℝ³
- Null(A) = {0}

**Example 2: 2×3 Matrix (Projection)**
```
A = [1 0 0]
    [0 1 0]
```
- Maps ℝ³ → ℝ²
- Col(A) = ℝ²
- Null(A) = Span{[0, 0, 1]ᵀ}

### Applications
- **Data compression:** Reducing dimensionality while preserving information
- **Computer graphics:** Projecting 3D objects onto 2D screens
- **Machine Learning:** Feature extraction, dimensionality reduction
- **Signal processing:** Sampling and reconstruction

---

## 9️⃣ Dot Products and Duality

### Dot Product (Inner Product)

The **dot product** of vectors v and w is:
```
v · w = v₁w₁ + v₂w₂ + ... + vₙwₙ = Σᵢ₌₁ⁿ vᵢwᵢ
```

#### Geometric Interpretation
```
v · w = |v||w|cos(θ)
```
where θ is the angle between the vectors.

#### Properties
- **Commutative:** v · w = w · v
- **Distributive:** u · (v + w) = u · v + u · w
- **Scalar multiplication:** (cv) · w = c(v · w)
- **Positive definite:** v · v ≥ 0, with equality only if v = 0

### Projection

The **projection** of vector v onto vector w is:
```
proj_w(v) = (v · w / |w|²)w = (v · ŵ)ŵ
```
where ŵ = w/|w| is the unit vector in the direction of w.

### Orthogonality

Vectors v and w are **orthogonal** if:
```
v · w = 0
```

### Duality

The **duality principle** states that every vector v can be viewed as:
1. **A geometric object:** a directed line segment
2. **A linear functional:** a function that maps other vectors to scalars

#### Vector as Linear Functional

For any vector v, we can define a linear functional:
```
f_v(w) = v · w
```

This functional:
- Maps ℝⁿ → ℝ
- Is linear: f_v(aw + bu) = af_v(w) + bf_v(u)
- Is bounded: |f_v(w)| ≤ |v||w|

#### Matrix Representation

The linear functional f_v can be represented as a row vector:
```
f_v(w) = vᵀw
```

### Examples

**Example 1: Computing Dot Product**
```
v = [3, 4], w = [1, 2]
v · w = 3·1 + 4·2 = 3 + 8 = 11
```

**Example 2: Projection**
```
v = [3, 4], w = [1, 0]
proj_w(v) = (3·1 + 4·0)/1² · [1, 0] = 3[1, 0] = [3, 0]
```

**Example 3: Orthogonality Check**
```
v = [1, -1], w = [1, 1]
v · w = 1·1 + (-1)·1 = 1 - 1 = 0 ✓ (orthogonal)
```

### Applications
- **Physics:** Work done by force, energy calculations
- **Computer Graphics:** Lighting models, shading
- **Machine Learning:** Similarity measures, feature selection
- **Signal Processing:** Correlation, filtering
- **Optimization:** Gradient descent, least squares

---

## 🔟 Cross Product

The **cross product** is a binary operation on vectors in $\mathbb{R}^3$ that produces a vector perpendicular to both input vectors.

### Definition

For vectors $\mathbf{v} = [v_1, v_2, v_3]^T$ and $\mathbf{w} = [w_1, w_2, w_3]^T$ in $\mathbb{R}^3$:
$$\mathbf{v} \times \mathbf{w} = \begin{bmatrix} v_2w_3 - v_3w_2 \\ v_3w_1 - v_1w_3 \\ v_1w_2 - v_2w_1 \end{bmatrix}$$

### Properties

#### Geometric Properties
- **Direction:** $\mathbf{v} \times \mathbf{w}$ is perpendicular to both $\mathbf{v}$ and $\mathbf{w}$
- **Magnitude:** $|\mathbf{v} \times \mathbf{w}| = |\mathbf{v}||\mathbf{w}|\sin(\theta)$
- **Right-hand rule:** Points in direction determined by right-hand rule

#### Algebraic Properties
- **Anticommutative:** $\mathbf{v} \times \mathbf{w} = -(\mathbf{w} \times \mathbf{v})$
- **Distributive:** $\mathbf{u} \times (\mathbf{v} + \mathbf{w}) = \mathbf{u} \times \mathbf{v} + \mathbf{u} \times \mathbf{w}$
- **Scalar multiplication:** $(c\mathbf{v}) \times \mathbf{w} = c(\mathbf{v} \times \mathbf{w}) = \mathbf{v} \times (c\mathbf{w})$
- **Self-product:** $\mathbf{v} \times \mathbf{v} = \mathbf{0}$

### Determinant Formula

The cross product can be computed using a determinant:
$$\mathbf{v} \times \mathbf{w} = \det\begin{pmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ v_1 & v_2 & v_3 \\ w_1 & w_2 & w_3 \end{pmatrix}$$

### Examples

**Example 1: Basic Cross Product**
$$\mathbf{v} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad \mathbf{w} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$$
$$\mathbf{v} \times \mathbf{w} = \begin{bmatrix} 0 \cdot 0 - 0 \cdot 1 \\ 0 \cdot 0 - 1 \cdot 0 \\ 1 \cdot 1 - 0 \cdot 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

**Example 2: Using Determinant**
$$\mathbf{v} = \begin{bmatrix} 2 \\ 3 \\ 4 \end{bmatrix}, \quad \mathbf{w} = \begin{bmatrix} 5 \\ 6 \\ 7 \end{bmatrix}$$
$$\mathbf{v} \times \mathbf{w} = \det\begin{pmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ 2 & 3 & 4 \\ 5 & 6 & 7 \end{pmatrix}$$
$$= \mathbf{i}(3 \cdot 7 - 4 \cdot 6) - \mathbf{j}(2 \cdot 7 - 4 \cdot 5) + \mathbf{k}(2 \cdot 6 - 3 \cdot 5)$$
$$= \mathbf{i}(21 - 24) - \mathbf{j}(14 - 20) + \mathbf{k}(12 - 15) = \begin{bmatrix} -3 \\ 6 \\ -3 \end{bmatrix}$$

### Applications

#### Area and Volume
- **Parallelogram area:** $|\mathbf{v} \times \mathbf{w}|$
- **Triangle area:** $\frac{1}{2}|\mathbf{v} \times \mathbf{w}|$
- **Parallelepiped volume:** $|\mathbf{u} \cdot (\mathbf{v} \times \mathbf{w})|$

#### Physics
- **Torque:** $\boldsymbol{\tau} = \mathbf{r} \times \mathbf{F}$
- **Angular momentum:** $\mathbf{L} = \mathbf{r} \times \mathbf{p}$
- **Magnetic force:** $\mathbf{F} = q(\mathbf{v} \times \mathbf{B})$

#### Computer Graphics
- **Surface normals:** Computing perpendicular vectors
- **Rotation axes:** Finding rotation directions
- **Orientation:** Determining handedness of coordinate systems

### Triple Products

#### Scalar Triple Product
$$\mathbf{u} \cdot (\mathbf{v} \times \mathbf{w}) = \det\begin{pmatrix} u_1 & u_2 & u_3 \\ v_1 & v_2 & v_3 \\ w_1 & w_2 & w_3 \end{pmatrix}$$

#### Vector Triple Product
$$\mathbf{u} \times (\mathbf{v} \times \mathbf{w}) = \mathbf{v}(\mathbf{u} \cdot \mathbf{w}) - \mathbf{w}(\mathbf{u} \cdot \mathbf{v})$$

---

## 11️⃣ Cross Product as Linear Transformation

The cross product can be expressed as a **matrix multiplication**, revealing its linear nature.

### Skew-Symmetric Matrix Representation

For any vector v = [v₁, v₂, v₃], we can define the **skew-symmetric matrix**:
```
[v]× = [ 0  -v₃  v₂]
       [ v₃  0  -v₁]
       [-v₂  v₁  0 ]
```

### Cross Product as Matrix Multiplication

The cross product v × w can be written as:
```
v × w = [v]× · w
```

### Properties of Skew-Symmetric Matrices

#### General Properties
- **[v]×ᵀ = -[v]×** (skew-symmetric)
- **[v]× · v = 0** (any vector crossed with itself is zero)
- **[v]× · w = -[w]× · v** (anticommutativity)

#### Eigenvalues and Eigenvectors
For [v]× where v ≠ 0:
- **Eigenvalue 0:** with eigenvector v (since [v]× · v = 0)
- **Eigenvalues ±i|v|:** with complex eigenvectors

### Examples

**Example 1: Basic Skew-Symmetric Matrix**
```
v = [1, 2, 3]
[v]× = [ 0  -3   2]
       [ 3   0  -1]
       [-2   1   0]
```

**Example 2: Cross Product via Matrix**
```
v = [1, 0, 0], w = [0, 1, 0]
[v]× = [ 0  0  0]
       [ 0  0 -1]
       [ 0  1  0]

v × w = [v]× · w = [ 0  0  0] [0]   [0]
                        [ 0  0 -1] [1] = [0]
                        [ 0  1  0] [0]   [1]
```

### Applications

#### Rotation Matrices
The skew-symmetric matrix is fundamental in rotation theory:
```
R = I + sin(θ)[û]× + (1-cos(θ))[û]×²
```
where û is a unit vector and θ is the rotation angle.

#### Lie Algebras
Skew-symmetric matrices form the Lie algebra so(3) of the rotation group SO(3).

#### Robotics
Used in:
- **Kinematics:** Representing angular velocities
- **Dynamics:** Moment of inertia tensors
- **Control:** Feedback linearization

### Connection to Quaternions

The skew-symmetric matrix representation connects to quaternion multiplication:
```
q₁q₂ = (s₁ + v₁)(s₂ + v₂) = s₁s₂ - v₁·v₂ + s₁v₂ + s₂v₁ + v₁×v₂
```
where the cross product term corresponds to the skew-symmetric part.

---

## 12️⃣ Cramer's Rule (Geometric View)

Cramer's Rule provides a geometric method for solving linear systems using determinants.

### The Rule

For the system Ax = b where A is an n×n invertible matrix:
```
xᵢ = det(Aᵢ) / det(A)
```
where Aᵢ is the matrix obtained by replacing the i-th column of A with the vector b.

### Geometric Interpretation

Each solution component xᵢ represents the **ratio of volumes** (or areas in 2D):
- **Numerator:** Volume of parallelepiped formed by replacing column i with b
- **Denominator:** Volume of parallelepiped formed by original columns of A

### Step-by-Step Process

1. **Check invertibility:** det(A) ≠ 0
2. **For each variable xᵢ:**
   - Replace column i of A with b to get Aᵢ
   - Compute det(Aᵢ)
   - Calculate xᵢ = det(Aᵢ) / det(A)

### Examples

**Example 1: 2×2 System**
```
2x + 3y = 7
4x + 5y = 11

A = [2 3], b = [7]
    [4 5]     [11]

det(A) = 2·5 - 3·4 = 10 - 12 = -2

A₁ = [7 3], det(A₁) = 7·5 - 3·11 = 35 - 33 = 2
     [11 5]

A₂ = [2 7], det(A₂) = 2·11 - 7·4 = 22 - 28 = -6
     [4 11]

x = det(A₁)/det(A) = 2/(-2) = -1
y = det(A₂)/det(A) = (-6)/(-2) = 3
```

**Example 2: 3×3 System**
```
x + 2y + 3z = 6
2x + 5y + 2z = 4
6x - 3y + z = 2

A = [1  2  3], b = [6]
    [2  5  2]     [4]
    [6 -3  1]     [2]

det(A) = 1(5·1 - 2·(-3)) - 2(2·1 - 2·6) + 3(2·(-3) - 5·6)
       = 1(5 + 6) - 2(2 - 12) + 3(-6 - 30)
       = 11 - 2(-10) + 3(-36)
       = 11 + 20 - 108 = -77

x = det(A₁)/det(A), y = det(A₂)/det(A), z = det(A₃)/det(A)
```

### Advantages and Disadvantages

#### Advantages
- **Geometric insight:** Direct connection to volumes/areas
- **Explicit formulas:** No iterative methods needed
- **Theoretical importance:** Shows relationship between determinants and solutions

#### Disadvantages
- **Computational cost:** O(n!) for n×n systems
- **Numerical instability:** For large matrices or near-singular systems
- **Limited applicability:** Only works for square, invertible systems

### Applications
- **Small systems:** 2×2 and 3×3 systems where geometric insight is valuable
- **Theoretical analysis:** Understanding the relationship between determinants and solutions
- **Educational purposes:** Demonstrating geometric principles in linear algebra

---

## 13️⃣ Change of Basis

Changing basis means expressing the same vector in different coordinate systems.

### Basis Change Matrix

If B = {v₁, v₂, ..., vₙ} is a new basis for ℝⁿ, the **change of basis matrix** is:
```
P = [v₁ v₂ ... vₙ]
```

### Coordinate Transformations

#### From New Basis to Standard Basis
```
x = P · x_B
```
where x_B are coordinates in the new basis.

#### From Standard Basis to New Basis
```
x_B = P⁻¹ · x
```

### Properties

1. **P is invertible** ⟺ {v₁, v₂, ..., vₙ} is a basis
2. **P⁻¹ exists** and represents the inverse transformation
3. **Composition:** If P₁: B₁ → B₂ and P₂: B₂ → B₃, then P₂P₁: B₁ → B₃

### Examples

**Example 1: Simple Basis Change**
```
Standard basis: e₁ = [1,0], e₂ = [0,1]
New basis: v₁ = [1,1], v₂ = [1,-1]

P = [1  1]
    [1 -1]

P⁻¹ = (1/2)[1  1]
           [1 -1]

Vector [3,4] in standard basis:
[3,4]_B = P⁻¹[3,4] = (1/2)[1  1][3] = (1/2)[7] = [3.5]
                          [1 -1][4]        [-1]   [-0.5]
```

**Example 2: Rotation Basis**
```
Standard basis: e₁ = [1,0], e₂ = [0,1]
Rotated basis: v₁ = [cos(θ),sin(θ)], v₂ = [-sin(θ),cos(θ)]

P = [cos(θ) -sin(θ)]
    [sin(θ)  cos(θ)]

P⁻¹ = Pᵀ (orthogonal matrix)
```

### Matrix Representation of Linear Transformations

If T has matrix A in standard basis, then in new basis B:
```
A_B = P⁻¹AP
```

### Applications

#### Diagonalization
If A has eigenvectors as columns of P, then:
```
P⁻¹AP = D (diagonal matrix)
```

#### Principal Component Analysis
- **Data transformation:** Rotating to principal axes
- **Dimensionality reduction:** Projecting onto important directions

#### Computer Graphics
- **Coordinate systems:** World, camera, screen coordinates
- **Transformations:** Rotating, scaling, translating objects

#### Quantum Mechanics
- **Representation theory:** Different bases for quantum states
- **Unitary transformations:** Preserving probabilities

### Orthogonal Change of Basis

When P is orthogonal (PᵀP = I):
- **Preserves lengths:** |Px| = |x|
- **Preserves angles:** (Px)·(Py) = x·y
- **Inverse is transpose:** P⁻¹ = Pᵀ

---

## 14️⃣ Eigenvectors and Eigenvalues

Eigenvectors and eigenvalues are fundamental concepts that reveal the "natural directions" of linear transformations.

### Definition

For a square matrix $A$, a nonzero vector $\mathbf{v}$ is an **eigenvector** with **eigenvalue** $\lambda$ if:
$$A\mathbf{v} = \lambda\mathbf{v}$$

### Characteristic Polynomial

The eigenvalues are roots of the **characteristic polynomial**:
$$p(\lambda) = \det(A - \lambda I) = 0$$

### Computing Eigenvalues

#### 2×2 Matrix
For $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$, the characteristic polynomial is:
$$\lambda^2 - (a + d)\lambda + (ad - bc) = 0$$

Using the quadratic formula:
$$\lambda = \frac{(a + d) \pm \sqrt{(a + d)^2 - 4(ad - bc)}}{2}$$

#### 3×3 Matrix
For $A = \begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix}$, expand $\det(A - \lambda I)$:
$$\det(A - \lambda I) = (a - \lambda)[(e - \lambda)(i - \lambda) - fh] - b[d(i - \lambda) - fg] + c[dh - (e - \lambda)g]$$

### Computing Eigenvectors

For each eigenvalue $\lambda_i$, solve:
$$(A - \lambda_i I)\mathbf{v} = \mathbf{0}$$

This gives a homogeneous system whose solution space is the **eigenspace** $E(\lambda_i)$.

### Examples

**Example 1: 2×2 Matrix**
$$A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$$

Characteristic polynomial: $\det\begin{pmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{pmatrix} = (3-\lambda)(2-\lambda) - 0 = 0$

$(3-\lambda)(2-\lambda) = 0$

$\lambda_1 = 3, \lambda_2 = 2$

For $\lambda_1 = 3$:
$$(A - 3I)\mathbf{v} = \begin{pmatrix} 0 & 1 \\ 0 & -1 \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$
$y = 0$, $x$ arbitrary
Eigenvector: $\mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$

For $\lambda_2 = 2$:
$$(A - 2I)\mathbf{v} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$
$x + y = 0$
Eigenvector: $\mathbf{v}_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$

**Example 2: Symmetric Matrix**
$$A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$$

Characteristic polynomial: $(2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = 0$
$\lambda_1 = 3, \lambda_2 = 1$

For $\lambda_1 = 3$: $\mathbf{v}_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$
For $\lambda_2 = 1$: $\mathbf{v}_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$

### Properties

#### General Properties
- **Sum of eigenvalues:** $\lambda_1 + \lambda_2 + \cdots + \lambda_n = \text{trace}(A)$
- **Product of eigenvalues:** $\lambda_1\lambda_2\cdots\lambda_n = \det(A)$
- **Eigenvalues of powers:** $A^k$ has eigenvalues $\lambda_1^k, \lambda_2^k, \ldots, \lambda_n^k$

#### Special Cases
- **Symmetric matrices:** All eigenvalues are real
- **Orthogonal matrices:** All eigenvalues have $|\lambda| = 1$
- **Triangular matrices:** Eigenvalues are diagonal entries

### Diagonalization

A matrix $A$ is **diagonalizable** if there exists an invertible matrix $P$ such that:
$$P^{-1}AP = D$$
where $D$ is diagonal with eigenvalues on the diagonal.

**Conditions for diagonalization:**
- $A$ has $n$ linearly independent eigenvectors
- $A$ has $n$ distinct eigenvalues (sufficient but not necessary)

### Applications

#### Principal Component Analysis (PCA)
- **Data analysis:** Finding principal directions of variation
- **Dimensionality reduction:** Projecting onto eigenvectors

#### Stability Analysis
- **Dynamical systems:** Eigenvalues determine system stability
- **Control theory:** Pole placement, system design

#### Quantum Mechanics
- **Energy levels:** Eigenvalues of Hamiltonian operator
- **Quantum states:** Eigenvectors represent stationary states

#### Computer Graphics
- **Principal axes:** Finding natural orientations of objects
- **Compression:** Reducing data dimensionality

### Numerical Methods

#### Power Method
For finding the largest eigenvalue:
```
vₖ₊₁ = Avₖ / |Avₖ|
```

#### QR Algorithm
Iterative method for finding all eigenvalues:
```
A₀ = A
Aₖ₊₁ = RₖQₖ (where Aₖ = QₖRₖ)
```

### Geometric Interpretation

- **Eigenvectors:** Directions that don't change under transformation
- **Eigenvalues:** Scaling factors along eigenvector directions
- **Eigenspace:** Subspace of all eigenvectors for a given eigenvalue

---

## 15️⃣ Abstract Vector Spaces

An **abstract vector space** generalizes the concept of vectors to any set of elements that satisfy the vector space axioms.

### Vector Space Axioms

A vector space V over a field F (usually ℝ or ℂ) is a set with two operations:
1. **Vector addition:** V × V → V
2. **Scalar multiplication:** F × V → V

**Axioms:**
1. **Closure:** u + v ∈ V, cv ∈ V
2. **Commutativity:** u + v = v + u
3. **Associativity:** (u + v) + w = u + (v + w)
4. **Identity:** ∃0 ∈ V such that v + 0 = v
5. **Inverse:** ∀v ∈ V, ∃(-v) ∈ V such that v + (-v) = 0
6. **Distributivity:** c(u + v) = cu + cv, (c + d)v = cv + dv
7. **Associativity:** (cd)v = c(dv)
8. **Identity:** 1v = v

### Examples of Vector Spaces

#### Function Spaces
**C[a,b]:** Continuous functions on [a,b]
- Addition: (f + g)(x) = f(x) + g(x)
- Scalar multiplication: (cf)(x) = cf(x)

**Pₙ:** Polynomials of degree ≤ n
- Basis: {1, x, x², ..., xⁿ}
- Dimension: n + 1

#### Matrix Spaces
**Mₘₓₙ:** m×n matrices
- Addition: (A + B)ᵢⱼ = Aᵢⱼ + Bᵢⱼ
- Scalar multiplication: (cA)ᵢⱼ = cAᵢⱼ
- Dimension: mn

#### Sequence Spaces
**ℓ²:** Square-summable sequences
- Elements: (a₁, a₂, a₃, ...) where Σ|aᵢ|² < ∞
- Inner product: ⟨a, b⟩ = Σaᵢb̄ᵢ

### Subspaces

A **subspace** W of V is a subset that is itself a vector space.

**Subspace Test:** W ⊆ V is a subspace if:
1. 0 ∈ W
2. u, v ∈ W ⟹ u + v ∈ W
3. u ∈ W, c ∈ F ⟹ cu ∈ W

### Examples of Subspaces

**Example 1: Polynomial Subspaces**
- P₂ ⊆ P₃ ⊆ P₄ ⊆ ...
- Even polynomials ⊆ Pₙ

**Example 2: Matrix Subspaces**
- Symmetric matrices ⊆ Mₙₓₙ
- Upper triangular matrices ⊆ Mₙₓₙ

### Linear Independence and Basis

#### Linear Independence
Vectors v₁, v₂, ..., vₖ are **linearly independent** if:
```
c₁v₁ + c₂v₂ + ... + cₖvₖ = 0 ⟹ c₁ = c₂ = ... = cₖ = 0
```

#### Basis
A **basis** for V is a linearly independent spanning set.

**Examples:**
- **P₂:** {1, x, x²}
- **M₂ₓ₂:** {[1 0], [0 1], [0 0], [0 0]}
              [0 0]  [0 0]  [1 0]  [0 1]

### Dimension

The **dimension** of V is the number of vectors in any basis.

**Examples:**
- dim(ℝⁿ) = n
- dim(Pₙ) = n + 1
- dim(Mₘₓₙ) = mn
- dim(C[a,b]) = ∞ (infinite-dimensional)

### Linear Transformations

A **linear transformation** T: V → W between vector spaces satisfies:
```
T(c₁v₁ + c₂v₂) = c₁T(v₁) + c₂T(v₂)
```

**Examples:**
- **Differentiation:** D: Pₙ → Pₙ₋₁
- **Integration:** I: C[a,b] → ℝ
- **Matrix multiplication:** T_A: ℝⁿ → ℝᵐ

### Applications

#### Signal Processing
- **Fourier analysis:** Functions as vectors in infinite-dimensional spaces
- **Filtering:** Linear transformations on signal spaces

#### Quantum Mechanics
- **State spaces:** Quantum states as vectors in Hilbert spaces
- **Operators:** Observables as linear transformations

#### Machine Learning
- **Feature spaces:** High-dimensional vector representations
- **Kernel methods:** Implicit mapping to infinite-dimensional spaces

#### Differential Equations
- **Solution spaces:** Sets of solutions form vector spaces
- **Linear operators:** Differential operators as linear transformations

---

## 🎯 Practice Problems

### Basic Vector Operations
1. **Compute the dot product:** $\mathbf{v} = [3, -2, 1]^T$, $\mathbf{w} = [1, 4, -3]^T$
2. **Find the cross product:** $\mathbf{v} = [2, 0, -1]^T$, $\mathbf{w} = [1, 3, 2]^T$
3. **Normalize the vector:** $\mathbf{v} = [4, 3, 0]^T$

### Linear Independence
4. **Check if vectors are independent:** $\mathbf{v}_1 = [1, 2, 3]^T$, $\mathbf{v}_2 = [2, 4, 6]^T$, $\mathbf{v}_3 = [1, 0, 1]^T$
5. **Find a basis for the span:** $\text{Span}\{[1, 1, 0]^T, [0, 1, 1]^T, [1, 0, 1]^T\}$

### Matrix Operations
6. **Compute the determinant:** $A = \begin{pmatrix} 3 & 1 & 2 \\ 0 & 4 & 1 \\ 2 & 0 & 3 \end{pmatrix}$
7. **Find the inverse:** $A = \begin{pmatrix} 2 & 1 \\ 3 & 2 \end{pmatrix}$

### Eigenvalues and Eigenvectors
8. **Find eigenvalues and eigenvectors:** $A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$
9. **Diagonalize the matrix:** $A = \begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix}$

### Applications
10. **PCA Problem:** Given data points $(1,2)$, $(2,4)$, $(3,6)$, find the principal components.

---

## 📚 Summary

This comprehensive guide covers all essential linear algebra concepts with detailed mathematical formulations, examples, and practical applications. Each topic connects directly to:

### Core Concepts
- **Vector Spaces**: Foundation for all linear algebra
- **Linear Transformations**: How matrices transform vectors
- **Eigenvalues/Eigenvectors**: Natural directions of transformations
- **Determinants**: Volume scaling and invertibility
- **Matrix Operations**: Computational foundations

### Key Applications
- **Machine Learning**: Feature spaces, dimensionality reduction, PCA
- **Computer Graphics**: 3D transformations, rotations, projections
- **Physics**: Quantum mechanics, coordinate transformations
- **Engineering**: Signal processing, structural analysis
- **Data Science**: Principal component analysis, linear regression

### Mathematical Tools
- **Matrix decompositions**: Eigenvalue decomposition, SVD
- **Linear systems**: Solving $A\mathbf{x} = \mathbf{b}$
- **Geometric interpretations**: Understanding transformations visually
- **Computational methods**: Numerical linear algebra

This mathematical foundation is crucial for understanding and implementing algorithms in artificial intelligence, machine learning, computer graphics, and scientific computing.

---

## 📚 Additional Resources

### Books
- **"Linear Algebra Done Right"** by Sheldon Axler
- **"Introduction to Linear Algebra"** by Gilbert Strang
- **"Linear Algebra and Its Applications"** by David Lay

### Online Resources
- **3Blue1Brown Linear Algebra Series** (YouTube)
- **Khan Academy Linear Algebra**
- **MIT OpenCourseWare 18.06**

### Software Tools
- **Python:** NumPy, SciPy, SymPy
- **MATLAB:** Built-in linear algebra functions
- **R:** Matrix package, eigen() function
- **Julia:** LinearAlgebra.jl package

