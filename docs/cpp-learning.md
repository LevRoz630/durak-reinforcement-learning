# C++ Concepts for the Durak Engine

Targeted learning path — only the C++ you need for this project.

## 1. Basic Types & Structs

How C++ represents data. You'll use this for `Card`, config values, etc.

- [learncpp.com — Fundamental data types](https://www.learncpp.com/cpp-tutorial/fundamental-data-types/) (~10min read)
- [learncpp.com — Structs](https://www.learncpp.com/cpp-tutorial/introduction-to-structs-members-and-member-selection/) (~10min read)
- [The Cherno — Structs in C++](https://www.youtube.com/watch?v=fLgTtaqqJp0) (8min video)

## 2. Enums

Perfect for suits, ranks, game phases — named constants with type safety.

- [learncpp.com — Scoped enumerations (enum class)](https://www.learncpp.com/cpp-tutorial/scoped-enumerations-enum-classes/) (~10min read)
- [The Cherno — ENUMS in C++](https://www.youtube.com/watch?v=x55jfOd5PEE) (7min video)

## 3. Classes

Bundling data + behavior. Your `Deck`, `GameState`, `Game` will be classes.

- [learncpp.com — Classes and class members](https://www.learncpp.com/cpp-tutorial/classes-and-class-members/) (~15min read)
- [learncpp.com — Public vs private access](https://www.learncpp.com/cpp-tutorial/public-vs-private-access-specifiers/) (~10min read)
- [The Cherno — Classes in C++](https://www.youtube.com/watch?v=2BP8NhxjrO0) (8min video)

## 4. Vectors (`std::vector`)

Dynamic arrays — used for hands, table cards, deck, legal actions.

- [learncpp.com — std::vector](https://www.learncpp.com/cpp-tutorial/introduction-to-stdvector-and-list-constructors/) (~10min read)
- [The Cherno — Vectors in C++](https://www.youtube.com/watch?v=PocJ80jmUGg) (13min video)

## 5. Header Files vs Source Files

How C++ organizes code across `.h` and `.cpp` files (no equivalent in Python).

- [learncpp.com — Header files](https://www.learncpp.com/cpp-tutorial/header-files/) (~10min read)
- [The Cherno — Header Files](https://www.youtube.com/watch?v=9RJTQmK0YPI) (12min video)

## 6. References & Const

Passing data efficiently without copying. Critical for performance.

- [learncpp.com — References](https://www.learncpp.com/cpp-tutorial/lvalue-references/) (~10min read)
- [learncpp.com — Const](https://www.learncpp.com/cpp-tutorial/const-variables-and-symbolic-constants/) (~10min read)
- [The Cherno — References](https://www.youtube.com/watch?v=IzoFn3dfsPA) (7min video)
- [The Cherno — Const](https://www.youtube.com/watch?v=4fJBrditnJU) (12min video)

## 7. Memory Model Basics

Stack vs heap — why C++ makes you think about where data lives (Python hides this).

- [learncpp.com — The stack and the heap](https://www.learncpp.com/cpp-tutorial/the-stack-and-the-heap/) (~10min read)
- [The Cherno — Stack vs Heap](https://www.youtube.com/watch?v=wJ1L2nSIV1s) (20min video)

## Later (pybind11 phase)

### 8. Namespaces

- [learncpp.com — Namespaces](https://www.learncpp.com/cpp-tutorial/user-defined-namespaces-and-the-scope-resolution-operator/) (~10min)

### 9. Build System & Compilation

- [The Cherno — How C++ Works](https://www.youtube.com/watch?v=SfGuIVzE_Os) (18min video — how .cpp becomes an executable)
- pybind11 handles the build for us via setuptools, so you don't need deep CMake knowledge

## Recommended Order

Go through 1-3 first, then come back and we'll build `Card` and `Deck` together. Pick up 4-7 as needed while building.

**Total time for 1-3: ~1.5 hours**
