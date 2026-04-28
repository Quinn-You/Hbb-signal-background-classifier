#include <iostream>
#include <TFile.h>
#include <TTree.h>
#include <TMVA/Factory.h>
#include <TMVA/DataLoader.h>
#include <TMVA/Tools.h>
#include <TMath.h>
#include <TString.h>

// 封装单次 Fold 的训练逻辑
// foldMod = 0 (Even), foldMod = 1 (Odd)
void run_bdt_training(int foldMod, TTree* tree) {
    TString tag = (foldMod == 0) ? "Even" : "Odd";
    std::cout << "\n\n" << std::string(60, '=') << std::endl;
    std::cout << "  正在启动 Fold: " << tag << " (EventNumber % 2 == " << foldMod << ")" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // 1. 创建输出文件和命名
    TString outputFileName = Form("TMVA_Output_%s.root", tag.Data());
    TFile* outFile = TFile::Open(outputFileName, "RECREATE");

    // 2. 初始化 Factory 和 DataLoader
    // 注意：DataLoader 的名字决定了输出文件夹的名字
    TMVA::Factory* factory = new TMVA::Factory(
        Form("TMVAClassification_%s", tag.Data()),
        outFile,
        "!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification"
    );

    TMVA::DataLoader* dataloader = new TMVA::DataLoader(Form("dataset_%s", tag.Data()));

    // 3. 添加变量
    dataloader->AddVariable("mBB", "Di-b-jet mass", "GeV", 'F');
    dataloader->AddVariable("dRBB", "#Delta R(b,b)", "", 'F');
    dataloader->AddVariable("pTV", "p_{T}^{V}", "GeV", 'F');
    dataloader->AddVariable("dPhiVBB", "#Delta#phi(V, bb)", "", 'F');
    dataloader->AddVariable("bin_MV2c10B2", "MV2c10 bin (b_{2})", "", 'F');
    dataloader->AddVariable("sumPtJets", "Sum p_{T} of jets", "GeV", 'F');

    // 4. 定义筛选条件：核心在于加入 EventNumber % 2
    TCut signalCut = Form("sample==\"qqWlvH125\" && nJ==2 && (EventNumber %% 2 == %d)", foldMod);
    TCut backgroundCut = Form("sample!=\"qqWlvH125\" && nJ==2 && (EventNumber %% 2 == %d)", foldMod);

    dataloader->AddSignalTree(tree, 1.0);
    dataloader->AddBackgroundTree(tree, 1.0);
    dataloader->SetSignalWeightExpression("EventWeight");
    dataloader->SetBackgroundWeightExpression("EventWeight");

    // 5. 准备训练和测试集
    // 由于我们要使用全量数据进行 Fold 训练，SplitMode 设为 Random，TMVA 自动划分训练/测试
    dataloader->PrepareTrainingAndTestTree(
        signalCut, backgroundCut,
        "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V"
    );

    // 6. 设定 BDT 参数 (使用你原代码中的配置)
    TString bdtOptions =
        "!H:!V:NTrees=800:MinNodeSize=2.5%:MaxDepth=4:BoostType=AdaBoost:"
        "AdaBoostBeta=0.3:UseBaggedBoost:BaggedSampleFraction=0.6:"
        "SeparationType=GiniIndex:nCuts=25:PruneMethod=NoPruning:"
        "UseYesNoLeaf=True:NegWeightTreatment=IgnoreNegWeightsInTraining";

    factory->BookMethod(dataloader, TMVA::Types::kBDT, "BDT_nJ2", bdtOptions);

    // 7. 运行
    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();

    // 8. 闭合文件
    outFile->Close();
    delete factory;
    delete dataloader;

    std::cout << "\n>>> Fold " << tag << " 训练完成！" << std::endl;
    std::cout << ">>> 权重文件位于: dataset_" << tag << "/weights/TMVAClassification_" << tag << "_BDT_nJ2.weights.xml" << std::endl;
}

void sumPtJets_plus() {
    TMVA::Tools::Instance();

    // 配置输入
    const char* inputFile = "/home/haoran/inputs.root";
    TFile* dataFile = TFile::Open(inputFile);
    if (!dataFile || dataFile->IsZombie()) {
        std::cerr << "错误：无法打开输入文件" << std::endl;
        return;
    }

    TTree* tree = (TTree*)dataFile->Get("Nominal");
    if (!tree) return;

    // 执行 2-Fold 训练
    // 1. 训练 Even 模型 (使用 EventNumber % 2 == 0)
    run_bdt_training(0, tree);

    // 2. 训练 Odd 模型 (使用 EventNumber % 2 == 1)
    run_bdt_training(1, tree);

    dataFile->Close();
    std::cout << "\n[成功] 所有 Fold 训练任务已完成。" << std::endl;
}