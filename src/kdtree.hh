#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "triangle.hh"

using iterator_v = std::vector<Triangle>::iterator;

class KdTree
{
public:
    struct KdNode
    {
        KdNode();
        KdNode(std::size_t& nodes_count, iterator_v beg, iterator_v end);

        std::unique_ptr<KdNode> left;
        std::unique_ptr<KdNode> right;

        float box[6]; // pair min : impair max
        iterator_v beg; // beg is median of axis
        iterator_v end; // end = beg + 1 if not leaf
        unsigned char axis = 0;

        void search(Ray &ray, double &dist) const;

        bool search_inter(const Ray &ray) const;

        inline bool is_child(void) const { return left == right; }

        unsigned size(void)
        {
            unsigned res = std::distance(beg, end);
            if (left)
                res += left.get()->size();
            if (right)
                res += right.get()->size();

            return res;
        }

        void print_infixe(void)
        {
            if (left)
                left.get()->print_infixe();

            std::cout << "extremum: ";
            for (unsigned i = 0; i < 6; ++i)
                std::cout << box[i] << " ";
            std::cout << '\n';
            for (auto it = beg; it < end; ++it)
            {
                std::cout << it->get_mean() << '\n';
            }
            if (right)
                right.get()->print_infixe();

            if (left)
                left.get()->print_infixe();

            if (right)
                right.get()->print_infixe();

        }
    };

    using childPtr = std::unique_ptr<KdNode>;

    KdTree(iterator_v beg, iterator_v end);
    bool search(Ray &r, double &dist) const
    {
        dist = -1;
        root_.get()->search(r, dist);

        return dist != -1;
    }

    void print_infixe()
    {
        root_.get()->print_infixe();
    }

    unsigned size(void)
    {
        return root_.get()->size();
    }

    childPtr root_;
    std::size_t nodes_count_ = 0;
private:
    static inline auto make_child()
    {
        return std::make_unique<KdNode>();
    }

    static inline auto make_child(std::size_t& nodes_count, iterator_v beg, iterator_v end)
    {
        ++nodes_count;
        return std::make_unique<KdNode>(nodes_count, beg, end);
    }

};
