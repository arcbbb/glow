#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/Optimizer.h"
#include "glow/Base/Type.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"
#include "gtest/gtest.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;

void SimpleHSplit(Function *F);
void SimpleVSplit(Function *F);
void SimpleVSplitSlow(Function *F);

MaxPoolNode *createMaxPool1(Function *F, Node *prev_node)
{
  ElemKind dataty = ElemKind::Int8QTy;
  Module *mod = F->getParent();
  MaxPoolNode *node_maxpool;
  std::vector<unsigned_t> kernel_hw{3, 3};
  std::vector<unsigned_t> stride_hw{2, 2};
  std::vector<unsigned_t> pad_tlbr{1, 1, 0, 0};
  std::array<size_t, 4> shape_output{1, 56, 56, 64};
  auto *type_output = mod->uniqueType(dataty, shape_output, 1, 0);
  node_maxpool = F->addNode(new MaxPoolNode("maxpool.1", type_output, prev_node,
        kernel_hw, stride_hw, pad_tlbr));
  return node_maxpool;
}


ConvolutionNode *createConv1(Function *F, Node *prev_node)
{
  ElemKind dataty = ElemKind::Int8QTy;
  ElemKind data32ty = ElemKind::Int32QTy;
  Module *mod = F->getParent();
  ConvolutionNode *node_conv;
  std::vector<unsigned_t> kernel_hw{7, 7};
  std::vector<unsigned_t> stride_hw{2, 2};
  std::vector<unsigned_t> pad_tlbr{3, 3, 3, 3};
  std::vector<unsigned_t> dilation_hw{1, 1};
  unsigned group = 1;

  // N, H, W, C
  std::array<size_t, 4> shape_output{1, 112, 112, 64};
  // OC, KH, KW, IC
  std::array<size_t, 4> shape_kernel{64, 7, 7, 3};
  std::array<size_t, 1> shape_bias{64};

  auto *node_filter = mod->createConstant(dataty, shape_kernel, 1, 0, "conv.f");
  auto *node_bias = mod->createConstant(data32ty, shape_bias, 1, 0, "conv.b");
  auto *type_output = mod->uniqueType(dataty, shape_output, 1, 0);

  node_conv = F->addNode(new ConvolutionNode("conv.1", type_output, prev_node,
        node_filter, node_bias, kernel_hw, stride_hw, pad_tlbr, group));

#if 0
  PseudoRNG PRNG;
  //node_filter->getHandle<int8_t>().initXavier(1, PRNG);
  node_filter->getHandle<int8_t>().randomize(0, 20, PRNG);
#else
  // Prepare conv filter data 64 x 3 x 7 x 7
  for (unsigned oc = 0; oc < 64; oc++) {
    for (unsigned ic = 0; ic < 3; ic++) {
      for (unsigned kh = 0; kh < 7; kh++) {
        for (unsigned kw = 0; kw < 7; kw++) {
          int val = 0;
          if (kh == 3 and kw == 3) {
            val = oc;
          }
          node_filter->getHandle<int8_t>().at({oc, kh, kw, ic}) = val;
        }
      }
    }
  }
#endif

  // Prepare conv bias data 64
  for (unsigned oc = 0; oc < 64; oc++) {
    node_bias->getHandle<int32_t>().at({oc}) = oc;
  }

  return node_conv;
}

std::pair<Function *, Tensor *> createAndInitFunction(Context &ctx,
                                                      ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("TestHIRConvSlice");

  ElemKind dataty = ElemKind::Int8QTy;
  std::array<size_t, 4> shape_input{1, 224, 224, 3};
  auto *node_input = mod.createPlaceholder(dataty, shape_input, 1, 0, "conv.in", false);
  auto *node_conv = createConv1(F, node_input);
  auto *node_maxpool = createMaxPool1(F, node_conv);
  auto *save = F->createSave("save", node_maxpool);
  auto *ctx_conv_in = ctx.allocate(node_input);
  auto *savePlaceholder = save->getPlaceholder();
  savePlaceholder->setName("save_output");
  auto *ctx_out = ctx.allocate(savePlaceholder);

#if 1
  PseudoRNG PRNG;
  //ctx_conv_in->getHandle<int8_t>().initXavier(1, PRNG);
  ctx_conv_in->getHandle<int8_t>().randomize(0, 10, PRNG);
#else
  // Prepare conv input data 1 x 3 x 224 x 224
  for (unsigned n = 0; n < 1; n++) {
    for (unsigned ic = 0; ic < 3; ic++) {
      for (unsigned ih = 0; ih < 224; ih++) {
        for (unsigned iw = 0; iw < 224; iw++) {
          int val = 3;
          ctx_conv_in->getHandle<int8_t>().at({n, ih, iw, ic}) = val;
        } // iw
      } // ih
    } // ic
  } // n
#endif

  return std::make_pair(F, ctx_out);
}

TEST(HIRSliceTest, ConvSliceData) {

  Context ctx1;
  Context ctx2;
  ExecutionEngine EE1{BackendKind::Interpreter};
  ExecutionEngine EE2{BackendKind::Interpreter};

  auto pair1 = createAndInitFunction(ctx1, EE1);
  auto pair2 = createAndInitFunction(ctx2, EE2);
  auto *F1 = pair1.first;
  auto *F2 = pair2.first;

  F1->dumpDAG("a3_origin.dot");
  SimpleHSplit(F1);
  F1->dumpDAG("a3_hsplit.dot");

  SimpleVSplitSlow(F2);
  F2->dumpDAG("a3_vsplit.dot");

#if 0
  auto *ctx_out1 = pair1.second;

  llvm::outs() << "First Run\n";
  EE1.compile(CompilationMode::Infer, F1);
  EE1.run(ctx1);

  llvm::outs() << "Second Run\n";
  EE2.compile(CompilationMode::Infer, F2);
  EE2.run(ctx2);

  pair1.second->isEqual(*pair2.second);
  auto H = ctx_out1->getHandle<int8_t>();
  H.dump();
#endif
}
