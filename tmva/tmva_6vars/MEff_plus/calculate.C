#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include "TMVA/Reader.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TString.h"

void calculate() {
    // ---------------------------------------------------------
    // 1. 初始化两个 TMVA Reader
    // ---------------------------------------------------------
    TMVA::Reader *readerEven = new TMVA::Reader("!Color:!Silent");
    TMVA::Reader *readerOdd  = new TMVA::Reader("!Color:!Silent");

    float mBB, dRBB, pTV, dPhiVBB,bin_MV2c10B2,MEff; // 变量必须是 float 类型，因为 TMVA Reader 只能处理 float
    // 必须把变量地址绑定给 Reader
    std::vector<TMVA::Reader*> readers = {readerEven, readerOdd};
    for(auto r : readers) {
        r->AddVariable("mBB", &mBB);
        r->AddVariable("dRBB", &dRBB);
        r->AddVariable("pTV", &pTV);
        r->AddVariable("dPhiVBB", &dPhiVBB);
        r->AddVariable("bin_MV2c10B2", &bin_MV2c10B2);
        r->AddVariable("MEff", &MEff);
    }

    // 使用你提供的绝对路径加载权重文件
    // 注意逻辑：Even模型用来测Odd数据，Odd模型用来测Even数据
    std::cout << "正在加载权重文件..." << std::endl;
    readerEven->BookMVA("BDT_Even", "/home/haoran/tmva/tmva_6vars/MEff_plus/dataset_Even/weights/TMVAClassification_Even_BDT_nJ2.weights.xml");
    readerOdd->BookMVA("BDT_Odd", "/home/haoran/tmva/tmva_6vars/MEff_plus/dataset_Odd/weights/TMVAClassification_Odd_BDT_nJ2.weights.xml");

    // ---------------------------------------------------------
    // 2. 加载数据文件
    // ---------------------------------------------------------
    TFile *input = TFile::Open("/home/haoran/inputs.root");
    if (!input || input->IsZombie()) {
        std::cerr << "错误：无法打开输入文件" << std::endl;
        return;
    }
    TTree *theTree = (TTree*)input->Get("Nominal");

    // ---------------------------------------------------------
    // 3. 绑定 Tree 变量 (注意类型匹配)
    // ---------------------------------------------------------
    float tree_mBB, tree_dRBB, tree_pTV, tree_dPhiVBB, tree_bin_MV2c10B2,tree_MEff;
    float eventWeight;
    int nJ;
    ULong64_t eventNumber; // EventNumber 在 ROOT 中通常是 64位无符号整数
    std::string *sampleName = 0;

    theTree->SetBranchAddress("mBB", &tree_mBB);
    theTree->SetBranchAddress("dRBB", &tree_dRBB);
    theTree->SetBranchAddress("pTV", &tree_pTV);
    theTree->SetBranchAddress("dPhiVBB", &tree_dPhiVBB);
    theTree->SetBranchAddress("bin_MV2c10B2", &tree_bin_MV2c10B2);
    theTree->SetBranchAddress("MEff", &tree_MEff);
    

    theTree->SetBranchAddress("nJ", &nJ);
    theTree->SetBranchAddress("EventWeight", &eventWeight);
    theTree->SetBranchAddress("sample", &sampleName);
    theTree->SetBranchAddress("EventNumber", &eventNumber);

    // ---------------------------------------------------------
    // 4. 定义直方图 (200 Bins)
    // ---------------------------------------------------------
    TH1F *h_sig = new TH1F("h_sig", "Signal (K-Fold Unbiased)", 200, -1, 1);
    TH1F *h_bkg = new TH1F("h_bkg", "Background (K-Fold Unbiased)", 200, -1, 1);

    // ---------------------------------------------------------
    // 5. 事件循环
    // ---------------------------------------------------------
    Long64_t nEntries = theTree->GetEntries();
    std::cout << "总计处理事例: " << nEntries << std::endl;

    for (Long64_t i = 0; i < nEntries; ++i) {
        theTree->GetEntry(i);

        if (nJ != 2) continue;

        // 变量映射到 Reader 绑定的地址
        mBB = tree_mBB;
        dRBB = tree_dRBB;
        pTV = tree_pTV;
        dPhiVBB = tree_dPhiVBB;
        bin_MV2c10B2 = tree_bin_MV2c10B2;
        MEff = tree_MEff;

        float score = -999.0;

        // --- 交叉评估 (Cross-Application) 逻辑 ---
        // 如果是偶数事例 (参与了 Even 模型训练)，则必须使用 Odd 模型来评估
        if (eventNumber % 2 == 0) {
            score = readerOdd->EvaluateMVA("BDT_Odd");
        } else {
            // 如果是奇数事例，则使用 Even 模型来评估
            score = readerEven->EvaluateMVA("BDT_Even");
        }

        double weight = (double)eventWeight;
        if (*sampleName == "qqWlvH125") h_sig->Fill(score, weight);
        else h_bkg->Fill(score, weight);

        if (i % 5000000 == 0) std::cout << "进度: " << (100.0 * i / nEntries) << "%" << std::endl;
    }

    // ---------------------------------------------------------
    // 6. 计算显著性
    // ---------------------------------------------------------
    double total_Z2 = 0;
    for (int i = 1; i <= 200; ++i) {
        double s = h_sig->GetBinContent(i);
        double b = h_bkg->GetBinContent(i);
        if (b > 1e-9 && s > 0) {
            total_Z2 += 2 * ((s + b) * std::log(1 + s / b) - s);
        }
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << ">>> 2-Fold 交叉验证显著性 Z = " << std::sqrt(total_Z2) << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    input->Close();
}