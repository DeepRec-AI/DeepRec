#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DICE_FUSION_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DICE_FUSION_H_

#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

//                                                         _______________
//                                                        |               |
//                                 _______________________|     INPUT     |________________________________
//                                |                       |               |                                |
//                                |                       |_______________|                                |
//                                |                               |                                        |
//                                |                            __\|/__                                     |
//                                |                          /         \                                   |
//                                |                         |           |                                  |
//                                |                         |   Sub_1   |                                  |
//                                |                         |           |                                  |
//                                |                          \_________/                                   |
//                                |                               |                                        |
//                                |                              \|/                                       |
//                                |                            _______               _____________         |
//                                |                          /         \            |             |        |
//                                |                         |           |           |             |        |
//                                |                         |   Mul_1   |/__________|    Rvar     |        |
//                                |                         |           |\          |             |        |
//                                |                          \_________/            |_____________|        |
//                                |                               |                                        |
//                                |                               |                                        |
//                                |      _______________         \|/                                       |
//                                |      |             |       _______                                     |
//                                |      |      1      |     /         \                                   |
//                                |      |  cosntant   |    |           |                                  |
//                                |      |_____________|    |  Sigmoid  |                                  |
//                                |              |          |           |                                  |
//                                |             \|/          \_________/                                   |
//                             __\|/__          _______           |                                        |
//                           /         \      /         \         |                                        |
//                          |           |    |           |        |                                        |
//                          |   Mul_2   |/___|   Sub_2   | /______|                                        |
//                          |           |\   |           | \                                               |
//                           \_________/      \_________/                                                  |
//                                |                                                                        |
//                                |                                                                        |
//                                |                                                                        |
//                                |                                                                        |
//                               \|/                                                                       |
//   _____________             _______                                                                  _______
//  |             |          /         \                                                              /         \
//  |             |         |           |                                                            |           |
//  |    Gamma    |________\|   Mul_3   |                                                            |   Mul_4   |
//  |             |        /|           |                                                            |           |
//  |_____________|          \_________/                                                              \_________/
//                                |                                                                        |
//                                |                                                                        |
//                                |                            _______                                     |
//                                |                          /         \                                   |
//                                |                         |           |                                  |
//                                |________________________\|    Add    |/_________________________________|
//                                                         /|           |\
//                                                           \_________/

// DiceFusion optimization for a graph.
class DiceFusion : public GraphOptimizer {
 public:
  DiceFusion() = default;
  explicit DiceFusion(RewriterConfig::Toggle opt_level) {}
  ~DiceFusion() override{};

  string name() const override { return "dice_fusion"; };

  bool UsesFunctionLibrary() const override { return false; }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result) override;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DICE_FUSION_H_
