#include "glow/Optimizer/Optimizer.h"
#include "glow/Base/Type.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/IR.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;

namespace {
class DebugDump {
public:
  static void dag(Function *F) {
    static DebugDump *p = new DebugDump();
    p->dumpdag(F);
  }
private:
  DebugDump() { d_c = 0; }
  void dumpdag(Function *F) {
    F->dumpDAG("debug" + std::to_string(d_c) + ".dot");
    d_c++;
  }
  int d_c;
};

MaxPoolNode *createNodeHelper(MaxPoolNode *node, TypeRef &Result, Node *Input,
                              llvm::ArrayRef<unsigned_t> Pads) {

  auto kernel_hw = node->getKernels();
  auto stride_hw = node->getStrides();
  auto *node_new = new MaxPoolNode(node->getName(), Result,
                                        Input, kernel_hw, stride_hw, Pads);
  Function *F = node->getParent();
  F->addNode(node_new);
  return node_new;
}

ConvolutionNode *createNodeHelper(ConvolutionNode *node, TypeRef &Result,
                                  Node *Input, llvm::ArrayRef<unsigned_t> Pads) {

  auto nodevalue_filter = node->getFilter();
  auto nodevalue_bias = node->getBias();
  auto kernel_hw = node->getKernels();
  auto stride_hw = node->getStrides();
  auto group = node->getGroup();
  auto *node_new = new ConvolutionNode(node->getName(), Result,
                                       Input, nodevalue_filter,
                                       nodevalue_bias, kernel_hw, stride_hw,
                                       Pads, group);
  Function *F = node->getParent();
  F->addNode(node_new);
  return node_new;
}

Node *getSliceOrigin(Node *node, std::vector<size_t> *StartOffset) {
  while (node->getKind() == Kinded::Kind::SliceNodeKind) {
    SliceNode *slice = llvm::cast<SliceNode>(node);
    auto parent_start = slice->getStart();
    std::transform(StartOffset->begin(), StartOffset->end(), parent_start.begin(),
                   StartOffset->begin(),std::plus<size_t>());
    node = slice->getInput();
  }
  return node;
}

template<typename T>
Node *createConvPoolSlice(T *node, llvm::ArrayRef<size_t> oslice_idx, llvm::ArrayRef<size_t> oslice_sz) {
  Function *F = node->getParent();
  // Assume Input & Result Shape: BatchSize, Dim0, Dim1, ..., Channel
  auto input_dim = node->getInput().getType()->dims();
  auto output_dim  = node->getResult().getType()->dims();
  ElemKind elemtype = node->getResult().getType()->getElementType();

  // Assume Kernel & Stride Shape:
  //   Dim0, Dim1, ...
  auto kernel_dim = node->getKernels();
  auto strides = node->getStrides();
  unsigned dim_num = kernel_dim.size();

  // Assume Padding Shape:
  //   Dim0_begin, Dim1_begin, ..., Dim0_end, Dim1_end, ...
  auto origin_pads = node->getPads();
  std::vector<unsigned_t> new_pads;
  new_pads.resize(dim_num * 2);

  std::vector<size_t> input_slice_dim;
  std::vector<size_t> start_offset;
  std::vector<size_t> oslice_shape;

  for (unsigned dim_idx = 0; dim_idx < dim_num + 2; dim_idx++) {

    unsigned oslice = oslice_sz[dim_idx];
    oslice_shape.push_back(oslice);

    if (oslice == output_dim[dim_idx]) {
      // No slice
      input_slice_dim.push_back(input_dim[dim_idx]);
      start_offset.push_back(0);
      if (dim_idx == 1 || dim_idx == 2) {
        new_pads[dim_idx - 1] = origin_pads[dim_idx -1];
        new_pads[dim_num + dim_idx - 1] = origin_pads[dim_num + dim_idx -1];
      }
      continue;
    }

    if (dim_idx == 0) {
      // Slice on BatchSize
      input_slice_dim.push_back(oslice);
      start_offset.push_back(oslice_idx[dim_idx]);
      continue;
    }

    if (dim_idx == dim_num + 2 - 1) {
      // Slice on Channel
      assert(false);
      // for Conv
      // input_slice_dim.push_back(input_dim[dim_idx]);
      // for Pooling
      // input_slice_dim.push_back(oslice);
    }

    unsigned kernel = kernel_dim[dim_idx - 1];
    unsigned stride = strides[dim_idx - 1];
    unsigned pad_begin = origin_pads[dim_idx - 1];

    unsigned output_idx = oslice_idx[dim_idx];
    unsigned input_idx = std::max(0, (int) (output_idx * stride - pad_begin));

    start_offset.push_back(input_idx);

    // Slice on Dimension

    // Calculate required input size
    // input_size = input_slice + padding
    unsigned input_size = (oslice - 1) * stride + kernel;

    unsigned pads = 0;
    // Calculate padding
    if (input_idx == 0) {
      unsigned cur_pad = pad_begin - output_idx * stride;
      new_pads[dim_idx - 1] = cur_pad;
      pads += cur_pad;
    }

    if ((input_idx + input_size - pads) > input_dim[dim_idx]) {
      unsigned cur_pad = input_idx + input_size - pads - input_dim[dim_idx];
      new_pads[dim_num + dim_idx] = cur_pad;
      pads += cur_pad;
    }

    // Calculate input_slice
    // input_slice = input_size - padding
    unsigned input_slice = input_size - pads;

    input_slice_dim.push_back(input_slice);
  }
  // Create slice
  auto type_islice = F->getParent()->uniqueType(elemtype, input_slice_dim, 1, 0);
  auto *orig_input = getSliceOrigin(node->getInput(), &start_offset);
  auto *node_input  = new SliceNode(orig_input->getName(), type_islice, orig_input, start_offset);
  F->addNode(node_input);

  // Dup Node
  auto *type_output = F->getParent()->uniqueType(elemtype, oslice_shape, 1, 0);
  auto *node_new = createNodeHelper(node, type_output, node_input, new_pads);
  return node_new;
}

template<typename T>
Node *createConvPoolSliceonDimH(T *node, unsigned oh_idx, unsigned oh_slice) {
  Function *F = node->getParent();
  auto dim_nhwc_i = node->getInput().getType()->dims();
  auto dim_nhwc_r = node->getResult().getType()->dims();
  ElemKind elemtype = node->getResult().getType()->getElementType();

  auto kernel_hw = node->getKernels();
  auto stride_hw = node->getStrides();

  unsigned batch_sz = dim_nhwc_i[0];
  unsigned ih = dim_nhwc_i[1];
  unsigned iw = dim_nhwc_i[2];
  unsigned ic = dim_nhwc_i[3];

  unsigned ow = dim_nhwc_r[2];
  unsigned oc = dim_nhwc_r[3];
  unsigned kh = kernel_hw[0];
  unsigned stride_h = stride_hw[0];

  auto padTLBR = node->getPads();
  std::vector<unsigned_t> pad_tlbr{1, padTLBR[1], 1, padTLBR[3]};

  // Caculate required input size
  // input_size = ih_slice + padding
  unsigned input_size = (oh_slice - 1) * stride_h + kh;

  unsigned pad_h_top = padTLBR[0];
  unsigned pad_h_bottom = padTLBR[2];
  unsigned ih_idx = std::max(0, (int) (oh_idx * stride_h - pad_h_top));

  // Calculate padding
  if (ih_idx == 0) {
      pad_tlbr[0] = pad_h_top;
  } else {
      pad_tlbr[0] = 0;
  }

  if ((ih_idx + input_size - pad_tlbr[0]) > ih) {
    pad_tlbr[2] = pad_h_bottom;
  } else {
    pad_tlbr[2] = 0;
  }

  // Calculate ih_slice
  // ih_slice = input_size - padding
  unsigned ih_slice = input_size - (pad_tlbr[0] + pad_tlbr[2]);

  // Create slices
  std::array<size_t, 4> islice_shape{batch_sz, ih_slice, iw, ic};
  std::array<size_t, 4> oslice_shape{batch_sz, oh_slice, ow, oc};

  auto type_islice = F->getParent()->uniqueType(elemtype, islice_shape, 1, 0);
  std::vector<size_t> start_offset{0, ih_idx, 0, 0};
  auto *orig_input = getSliceOrigin(node->getInput(), &start_offset);
  auto *node_input  = new SliceNode(orig_input->getName(), type_islice, orig_input, start_offset);
  F->addNode(node_input);
  auto *type_output = F->getParent()->uniqueType(elemtype, oslice_shape, 1, 0);
  auto *node_new = createNodeHelper(node, type_output, node_input, pad_tlbr);
  return node_new;
}

#if 1
Node *createSlice(Node *node, llvm::ArrayRef<size_t> oslice_idx, llvm::ArrayRef<size_t> oslice_sz) {
  switch (node->getKind()) {
    case Kinded::Kind::ConvolutionNodeKind:
      {
        auto *n = llvm::cast<ConvolutionNode>(node);
        return createConvPoolSliceonDimH(n, oslice_idx[1], oslice_sz[1]);
      }
      break;
    case Kinded::Kind::MaxPoolNodeKind:
      {
        auto *n = llvm::cast<MaxPoolNode>(node);
        return createConvPoolSliceonDimH(n, oslice_idx[1], oslice_sz[1]);
      }
      break;
    default:
      llvm::outs() << node->getKindName() << "\n";
      llvm_unreachable("node is not support");
  }
  return nullptr;
}
#else
Node *createSlice(Node *node, llvm::ArrayRef<size_t> oslice_idx, llvm::ArrayRef<size_t> oslice_sz) {
  switch (node->getKind()) {
    case Kinded::Kind::ConvolutionNodeKind:
      {
        auto *n = llvm::cast<ConvolutionNode>(node);
        return createConvPoolSlice(n, oslice_idx, oslice_sz);
      }
      break;
    case Kinded::Kind::MaxPoolNodeKind:
      {
        auto *n = llvm::cast<MaxPoolNode>(node);
        return createConvPoolSlice(n, oslice_idx, oslice_sz);
      }
      break;
    default:
      llvm::outs() << node->getKindName() << "\n";
      llvm_unreachable("node is not support");
  }
  return nullptr;
}
#endif

Node *createSliceonDimH(Node *node, unsigned oh_idx, unsigned oh_slice) {
  switch (node->getKind()) {
    case Kinded::Kind::ConvolutionNodeKind:
      {
        auto *n = llvm::cast<ConvolutionNode>(node);
        return createConvPoolSliceonDimH(n, oh_idx, oh_slice);
      }
      break;
    case Kinded::Kind::MaxPoolNodeKind:
      {
        auto *n = llvm::cast<MaxPoolNode>(node);
        return createConvPoolSliceonDimH(n, oh_idx, oh_slice);
      }
      break;
    default:
      llvm::outs() << node->getKindName() << "\n";
      llvm_unreachable("node is not support");
  }
  return nullptr;
}

template<typename T>
void DupNodeForSlice(T *node) {

  if (node->getKind() == Kinded::Kind::SliceNodeKind)
    return;

  for (auto it : node->getUsers()) {
    auto *user = it.getUser();
    if (auto *slice = llvm::dyn_cast<SliceNode>(user)) {
      auto dim_slice = slice->getResult().getType()->dims();
      auto vec_start = slice->getStart();

      auto *new_node = createSlice(node, vec_start, dim_slice);

      // Replace Slice's users with new node
      NodeValue(slice, 0).replaceAllUsesOfWith(new_node);
    }
  }
}

class SliceParam {
public:
  SliceParam(Node *node) {
    dim_h_ = 0;
    max_slice_h_ = 0;
    is_supported_ = false;
    init(node);
  }

  void init(Node *node) {
    switch (node->getKind()) {
      case Kinded::Kind::ConvolutionNodeKind:
        {
          auto *n = llvm::cast<ConvolutionNode>(node);
          init(n);
        }
        break;
      case Kinded::Kind::MaxPoolNodeKind:
        {
          auto *n = llvm::cast<MaxPoolNode>(node);
          init(n);
        }
        break;
    }
  }

  void init(ConvolutionNode *node) {
    auto dim_nhwc_r = node->getResult().getType()->dims();
    dim_h_ = dim_nhwc_r[1];
    max_slice_h_ = 56; // for demonstration only
    is_supported_ = true;
  }

  void init(MaxPoolNode *node) {
    auto dim_nhwc_r = node->getResult().getType()->dims();
    dim_h_ = dim_nhwc_r[1];
    max_slice_h_ = 28; // for demonstration only
    is_supported_ = true;
  }

  bool isSupported() { return is_supported_; }

  unsigned getDimH() { return dim_h_; }
  unsigned getMaxHSlice() { return max_slice_h_; }

private:
  bool is_supported_;
  unsigned dim_h_;
  unsigned max_slice_h_;
};

void sliceEqually(Node *node, unsigned oh, unsigned max_h_slice) {
  unsigned h_idx = 0;
  std::vector<NodeValue> concat_list;

#if 0
  std::vector<std::vector<NodeValue>> concat_multi;
  while (check_last_offset) {
    NodeValue nv = createSlice(node, dim_offset, max_slice);
    concat_multi[[0].push_back(nv);
    // update offset
    bool carry = true;
    for (dim_idx = 0; carry && dim_idx < dim_num; dim_idx++) {
      dim_offset[dim_idx] += max_slice[dim_idx];
      dim_sz[dim_idx] -= max_slice[dim_idx];
      if (dim_sz[dim_idx] > max_slice[dim_idx]) {
        dim_offset[dim_idx] = 0;
        carry = true;
        // Node * nv= createConcat(concat_multi[dim_idx]);
        // concat_multi[dim_idx+1].push_back(nv);
      } else {
        carry = false;
      }
    }
  }

  for (dim_idx = 0; dim_idx < dim_num; dim_idx++) {
    while (oslice > max_slice) {
      concat_list.push_back(createSliceonDimH(node, dim_offset, max_slice));
      dim_offset += max_slice;
      oslice -= max_slice;
    }
    concat_list.push_back(createSliceonDimH(node, dim_offset, oslice));
    auto *node_concat = F->createConcat(node->getName(), concat_list, dim_idx);
  }
#endif
  assert(oh > max_h_slice);
  while (oh > max_h_slice) {
    concat_list.push_back(createSliceonDimH(node, h_idx, max_h_slice));
    h_idx += max_h_slice;
    oh -= max_h_slice;
  }

  concat_list.push_back(createSliceonDimH(node, h_idx, oh));
  int index_h = 1;
  Function *F = node->getParent();
  auto *node_concat = F->createConcat(node->getName(), concat_list, index_h);
  int result_index = 0;

  DebugDump::dag(F);
  NodeValue(node, result_index).replaceAllUsesOfWith(node_concat);
  DebugDump::dag(F);
}

bool trySliceEqually(Node *node) {
  SliceParam param(node);
  unsigned oh = param.getDimH();
  unsigned max_h_slice = param.getMaxHSlice();

  if ((param.isSupported() == false) || (node->getNumUsers() == 0)
      || (oh <= max_h_slice) || ((oh - max_h_slice) < max_h_slice))
    return false;

  sliceEqually(node, oh, max_h_slice);
  return true;
}

std::vector<SaveNode *> getSaveNodeList(Function *F) {
  std::vector<SaveNode *> list_save;
  for (auto &node : F->getNodes()) {
    auto *n = llvm::dyn_cast<SaveNode>(&node);
    if (n)
      list_save.push_back(n);
  }
  return list_save;
}
} // namespace

void SimpleHSplit(Function *F) {
  auto &nodes = F->getNodes();
  for (auto it = nodes.begin(), e = nodes.end(); it != e; it++) {
    Node *nodep = &*it;
    trySliceEqually(nodep);
  }
  ::glow::optimize(F, CompilationMode::Infer);
}

void SimpleVSplitSlow(Function *F) {
  auto &nodes = F->getNodes();
  while (true) {
    bool changed = false;
    for (auto it = nodes.rbegin(), e = nodes.rend(); it != e; it++) {
      Node *nodep = &*it;
      if (changed) {
        DupNodeForSlice(nodep);
      } else {
        changed = trySliceEqually(nodep);
      }
    }
    if (changed == false) {
      break;
    }
    ::glow::optimize(F, CompilationMode::Infer);
  }
}

void SimpleVSplit(Function *F) {
  std::vector<SaveNode *> list_save = getSaveNodeList(F);
  for (auto *save : list_save) {
    Node *node = save->getInput().getNode();
    llvm::outs() << node->getName() << "\n";
    bool changed = trySliceEqually(node);
    if (changed) {
    }
  }
}
