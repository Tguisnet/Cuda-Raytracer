#include <algorithm>
#include <array>
#ifdef __GNUC__
#include <parallel/algorithm>
#endif
#include <iostream>
#include <iterator>
#include <omp.h>

#include "kdtree.hh"

#define MIN_TRIANGLES 4

const std::function<bool (const Triangle&, const Triangle&)> func[3] =
{
    [](const Triangle &a, const Triangle &b) { return a.get_mean()[0] < b.get_mean()[0]; },
    [](const Triangle &a, const Triangle &b) { return a.get_mean()[1] < b.get_mean()[1]; },
    [](const Triangle &a, const Triangle &b) { return a.get_mean()[2] < b.get_mean()[2]; }
};

static void get_extremum(float box[6], iterator_v beg,
                                       iterator_v end)
{
    box[0] = beg->vertices[0][0];
    box[1] = beg->vertices[0][0];

    box[2] = beg->vertices[0][1];
    box[3] = beg->vertices[0][1];

    box[4] = beg->vertices[0][2];
    box[5] = beg->vertices[0][2];

    while (beg < end)
    {
        for (unsigned i = 0; i < 3; ++i)
        {
            box[0] = std::min(box[0], beg->vertices[i][0]);
            box[1] = std::max(box[1], beg->vertices[i][0]);

            box[2] = std::min(box[2], beg->vertices[i][1]);
            box[3] = std::max(box[3], beg->vertices[i][1]);

            box[4] = std::min(box[4], beg->vertices[i][2]);
            box[5] = std::max(box[5], beg->vertices[i][2]);
        }
        ++beg;
    }

    for (unsigned i = 0; i < 6; i += 2) // expand bounding box
        box[i] -= EPSILON;
    for (unsigned i = 1; i < 6; i += 2)
        box[i] += EPSILON;
}

static unsigned get_longest_axis(const float box[6])
{
    const std::array<float, 3> diffs = {
        box[1] - box[0],
        box[3] - box[2],
        box[5] - box[4]
    };

    return std::distance(
            diffs.begin(),
            std::max_element(diffs.begin(), diffs.end())
            );
}

KdTree::KdTree(iterator_v beg, iterator_v end)
{
    root_ = make_child(nodes_count_, beg, end);
}

KdTree::KdNode::KdNode(std::size_t& nodes_count, iterator_v beg, iterator_v end)
{
    unsigned dist = std::distance(beg, end);
    get_extremum(box, beg, end);

    if (dist < MIN_TRIANGLES)
    {
        this->beg = beg;
        this->end = end;
        left = nullptr;
        right = nullptr;

        return;
    }

    axis = get_longest_axis(box);
#ifdef __GNUC__
    // Takes advantage of GNU Parallel STL
    __gnu_parallel::sort(beg, end, func[axis]);
#else
    sort(beg, end, func[axis]);
#endif

    const auto med = beg + dist / 2;

    left = make_child(nodes_count, beg, med);
    right = make_child(nodes_count, med + 1, end);

    this->beg = med;
    this->end = med + 1;
}
