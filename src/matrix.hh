#pragma once

#include <vector>
#include <iostream>
#include <initializer_list>

#include "vector.hh"

class Matrix
{
    public:
        Matrix() = default;
        Matrix(unsigned i, unsigned j);
        Matrix(unsigned i, unsigned j, std::initializer_list<float> vals);

        Matrix operator*(const Matrix&) const;
        Matrix mat_mul(const Matrix&) const;

        friend Vector operator*(const Vector& vec, const Matrix& mat);

        float& operator[](unsigned pos);
        float operator[](unsigned pos) const;

        friend std::ostream& operator <<(std::ostream& os, 
                                         const Matrix& mat)
        {
            for (unsigned i = 0; i < mat.lines_; ++i)
            {
                for (unsigned j = 0; j < mat.cols_; ++j)
                {
                    os << mat[i * mat.cols_ + j] << " | ";
                }
                os << std::endl;
            }

            return os;
        }

    private:
        unsigned lines_;
        unsigned cols_;
        std::vector<float> content_;
};
