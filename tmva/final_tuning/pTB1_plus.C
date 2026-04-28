//root -l -b -q 'pTB1_plus.C(true)'


#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <TFile.h>
#include <TMath.h>
#include <TString.h>
#include <TSystem.h>
#include <TTree.h>
#include <TMVA/DataLoader.h>
#include <TMVA/Factory.h>
#include <TMVA/Reader.h>
#include <TMVA/Tools.h>
#include <TH1F.h>

struct BDTConfig {
    int nTrees;
    int maxDepth;
    double adaBoostBeta;
};

struct FoldArtifacts {
    TString foldTag;
    TString datasetDir;
    TString factoryName;
    TString weightFile;
    TString outputRootFile;
};

TString sanitize_beta(double value) {
    TString betaText = Form("%.2f", value);
    betaText.ReplaceAll(".", "p");
    return betaText;
}

TString config_tag(const BDTConfig& config) {
    return Form(
        "NTrees%d_MaxDepth%d_Beta%s",
        config.nTrees,
        config.maxDepth,
        sanitize_beta(config.adaBoostBeta).Data()
    );
}

TString build_scan_dir(const BDTConfig& config) {
    return TString("grid_scan_output/") + config_tag(config);
}

void add_training_variables(TMVA::DataLoader* dataloader) {
    dataloader->AddVariable("mBB", "Di-b-jet mass", "GeV", 'F');
    dataloader->AddVariable("dRBB", "#Delta R(b,b)", "", 'F');
    dataloader->AddVariable("pTV", "p_{T}^{V}", "GeV", 'F');
    dataloader->AddVariable("dPhiVBB", "#Delta#phi(V, bb)", "", 'F');
    dataloader->AddVariable("bin_MV2c10B2", "MV2c10 bin (b_{2})", "", 'F');
    dataloader->AddVariable("MET", "E_{T}^{miss}", "GeV", 'F');
    dataloader->AddVariable("bin_MV2c10B1", "MV2c10 bin (b_{1})", "", 'F');
    dataloader->AddVariable("pTB2", "p_{T}^{B2}", "GeV", 'F');
    dataloader->AddVariable("pTB1", "p_{T}^{B1}", "GeV", 'F');
}

void bind_reader_variables(
    TMVA::Reader* reader,
    float& mBB,
    float& dRBB,
    float& pTV,
    float& dPhiVBB,
    float& bin_MV2c10B2,
    float& MET,
    float& bin_MV2c10B1,
    float& pTB2,
    float& pTB1
) {
    reader->AddVariable("mBB", &mBB);
    reader->AddVariable("dRBB", &dRBB);
    reader->AddVariable("pTV", &pTV);
    reader->AddVariable("dPhiVBB", &dPhiVBB);
    reader->AddVariable("bin_MV2c10B2", &bin_MV2c10B2);
    reader->AddVariable("MET", &MET);
    reader->AddVariable("bin_MV2c10B1", &bin_MV2c10B1);
    reader->AddVariable("pTB2", &pTB2);
    reader->AddVariable("pTB1", &pTB1);
}

FoldArtifacts run_bdt_training(const BDTConfig& config, int foldMod, TTree* tree) {
    TString foldTag = (foldMod == 0) ? "Even" : "Odd";
    TString scanDir = build_scan_dir(config);
    TString datasetDir = scanDir + Form("/dataset_%s", foldTag.Data());
    TString factoryName = Form("TMVAClassification_%s_%s", foldTag.Data(), config_tag(config).Data());
    TString outputRootFile = scanDir + Form("/TMVA_Output_%s.root", foldTag.Data());
    TString weightFile = datasetDir + Form("/weights/%s_BDT.weights.xml", factoryName.Data());

    gSystem->mkdir(scanDir, kTRUE);
    gSystem->mkdir(datasetDir, kTRUE);

    std::cout << "\n\n" << std::string(72, '=') << std::endl;
    std::cout << "开始训练 Fold: " << foldTag
              << " | " << config_tag(config)
              << " | EventNumber % 2 == " << foldMod << std::endl;
    std::cout << std::string(72, '=') << std::endl;

    TFile* outFile = TFile::Open(outputRootFile, "RECREATE");
    if (!outFile || outFile->IsZombie()) {
        throw std::runtime_error(std::string("无法创建输出文件: ") + outputRootFile.Data());
    }

    TMVA::Factory* factory = new TMVA::Factory(
        factoryName,
        outFile,
        "!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification"
    );

    TMVA::DataLoader* dataloader = new TMVA::DataLoader(datasetDir);
    add_training_variables(dataloader);

    TCut signalCut = Form("sample==\"qqWlvH125\" && nJ==2 && (EventNumber %% 2 == %d)", foldMod);
    TCut backgroundCut = Form("sample!=\"qqWlvH125\" && nJ==2 && (EventNumber %% 2 == %d)", foldMod);

    dataloader->AddSignalTree(tree, 1.0);
    dataloader->AddBackgroundTree(tree, 1.0);
    dataloader->SetSignalWeightExpression("EventWeight");
    dataloader->SetBackgroundWeightExpression("EventWeight");
    dataloader->PrepareTrainingAndTestTree(
        signalCut,
        backgroundCut,
        "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V"
    );

    TString bdtOptions = Form(
        "!H:!V:NTrees=%d:MinNodeSize=2.5%%:MaxDepth=%d:BoostType=AdaBoost:"
        "AdaBoostBeta=%.3f:UseBaggedBoost:BaggedSampleFraction=0.6:"
        "SeparationType=GiniIndex:nCuts=25:PruneMethod=NoPruning:"
        "UseYesNoLeaf=True:NegWeightTreatment=IgnoreNegWeightsInTraining",
        config.nTrees,
        config.maxDepth,
        config.adaBoostBeta
    );

    factory->BookMethod(dataloader, TMVA::Types::kBDT, "BDT", bdtOptions);
    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();

    outFile->Close();
    delete factory;
    delete dataloader;
    delete outFile;

    std::cout << "Fold " << foldTag << " 训练完成，权重文件: " << weightFile << std::endl;

    FoldArtifacts artifacts;
    artifacts.foldTag = foldTag;
    artifacts.datasetDir = datasetDir;
    artifacts.factoryName = factoryName;
    artifacts.weightFile = weightFile;
    artifacts.outputRootFile = outputRootFile;
    return artifacts;
}

double calculate_significance(TTree* tree, const FoldArtifacts& evenArtifacts, const FoldArtifacts& oddArtifacts) {
    TMVA::Reader* readerEven = new TMVA::Reader("!Color:!Silent");
    TMVA::Reader* readerOdd = new TMVA::Reader("!Color:!Silent");

    float mBB, dRBB, pTV, dPhiVBB, bin_MV2c10B2, MET, bin_MV2c10B1, pTB2, pTB1;
    bind_reader_variables(readerEven, mBB, dRBB, pTV, dPhiVBB, bin_MV2c10B2, MET, bin_MV2c10B1, pTB2, pTB1);
    bind_reader_variables(readerOdd, mBB, dRBB, pTV, dPhiVBB, bin_MV2c10B2, MET, bin_MV2c10B1, pTB2, pTB1);

    readerEven->BookMVA("BDT", evenArtifacts.weightFile);
    readerOdd->BookMVA("BDT", oddArtifacts.weightFile);

    float tree_mBB, tree_dRBB, tree_pTV, tree_dPhiVBB, tree_bin_MV2c10B2;
    float tree_MET, tree_bin_MV2c10B1, tree_pTB2, tree_pTB1;
    float eventWeight;
    int nJ;
    ULong64_t eventNumber;
    std::string* sampleName = 0;

    tree->SetBranchAddress("mBB", &tree_mBB);
    tree->SetBranchAddress("dRBB", &tree_dRBB);
    tree->SetBranchAddress("pTV", &tree_pTV);
    tree->SetBranchAddress("dPhiVBB", &tree_dPhiVBB);
    tree->SetBranchAddress("bin_MV2c10B2", &tree_bin_MV2c10B2);
    tree->SetBranchAddress("MET", &tree_MET);
    tree->SetBranchAddress("bin_MV2c10B1", &tree_bin_MV2c10B1);
    tree->SetBranchAddress("pTB2", &tree_pTB2);
    tree->SetBranchAddress("pTB1", &tree_pTB1);
    tree->SetBranchAddress("nJ", &nJ);
    tree->SetBranchAddress("EventWeight", &eventWeight);
    tree->SetBranchAddress("sample", &sampleName);
    tree->SetBranchAddress("EventNumber", &eventNumber);

    TH1F* h_sig = new TH1F("h_sig", "Signal (K-Fold Unbiased)", 200, -1, 1);
    TH1F* h_bkg = new TH1F("h_bkg", "Background (K-Fold Unbiased)", 200, -1, 1);

    Long64_t nEntries = tree->GetEntries();
    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        if (nJ != 2) {
            continue;
        }

        mBB = tree_mBB;
        dRBB = tree_dRBB;
        pTV = tree_pTV;
        dPhiVBB = tree_dPhiVBB;
        bin_MV2c10B2 = tree_bin_MV2c10B2;
        MET = tree_MET;
        bin_MV2c10B1 = tree_bin_MV2c10B1;
        pTB2 = tree_pTB2;
        pTB1 = tree_pTB1;

        float score = (eventNumber % 2 == 0)
            ? readerOdd->EvaluateMVA("BDT")
            : readerEven->EvaluateMVA("BDT");

        double weight = static_cast<double>(eventWeight);
        if (*sampleName == "qqWlvH125") {
            h_sig->Fill(score, weight);
        } else {
            h_bkg->Fill(score, weight);
        }

        if (i > 0 && i % 5000000 == 0) {
            std::cout << "显著性计算进度: " << (100.0 * i / nEntries) << "%" << std::endl;
        }
    }

    double totalZ2 = 0.0;
    for (int i = 1; i <= h_sig->GetNbinsX(); ++i) {
        double s = h_sig->GetBinContent(i);
        double b = h_bkg->GetBinContent(i);
        if (b > 1e-9 && s > 0.0) {
            totalZ2 += 2.0 * ((s + b) * std::log(1.0 + s / b) - s);
        }
    }

    double significance = std::sqrt(totalZ2);

    delete h_sig;
    delete h_bkg;
    delete readerEven;
    delete readerOdd;
    tree->ResetBranchAddresses();

    return significance;
}

double run_single_config(TTree* tree, const BDTConfig& config) {
    FoldArtifacts evenArtifacts = run_bdt_training(config, 0, tree);
    FoldArtifacts oddArtifacts = run_bdt_training(config, 1, tree);
    double significance = calculate_significance(tree, evenArtifacts, oddArtifacts);

    std::cout << std::string(72, '-') << std::endl;
    std::cout << "配置 " << config_tag(config) << " 的 2-Fold significance = " << significance << std::endl;
    std::cout << std::string(72, '-') << std::endl;

    return significance;
}

void run_grid_scan(TTree* tree) {
    std::vector<int> nTreesGrid = {400, 800, 1200};
    std::vector<int> maxDepthGrid = {3, 4, 5};
    std::vector<double> adaBoostBetaGrid = {0.2, 0.3, 0.5};

    std::ofstream summaryFile("grid_scan_results.csv");
    summaryFile << "NTrees,MaxDepth,AdaBoostBeta,Significance\n";

    double bestSignificance = -std::numeric_limits<double>::infinity();
    BDTConfig bestConfig = {0, 0, 0.0};
    int totalConfigs = nTreesGrid.size() * maxDepthGrid.size() * adaBoostBetaGrid.size();
    int configIndex = 0;

    for (std::size_t i = 0; i < nTreesGrid.size(); ++i) {
        for (std::size_t j = 0; j < maxDepthGrid.size(); ++j) {
            for (std::size_t k = 0; k < adaBoostBetaGrid.size(); ++k) {
                ++configIndex;
                BDTConfig config = {nTreesGrid[i], maxDepthGrid[j], adaBoostBetaGrid[k]};
                
                if (configIndex <= 24) {
                std::cout << "\n[Grid " << configIndex << "/" << totalConfigs 
                          << "] 跳过已完成配置: " << config_tag(config) << std::endl;
                    continue;
                }

                std::cout << "\n[Grid " << configIndex << "/" << totalConfigs << "] 扫描配置: "
                          << config_tag(config) << std::endl;

                double significance = run_single_config(tree, config);
                summaryFile << config.nTrees << ","
                            << config.maxDepth << ","
                            << std::fixed << std::setprecision(3) << config.adaBoostBeta << ","
                            << std::setprecision(6) << significance << "\n";

                if (significance > bestSignificance) {
                    bestSignificance = significance;
                    bestConfig = config;
                }
            }
        }
    }

    summaryFile.close();

    std::ofstream bestFile("grid_scan_best.txt");
    bestFile << "BestConfigTag=" << config_tag(bestConfig).Data() << "\n";
    bestFile << "NTrees=" << bestConfig.nTrees << "\n";
    bestFile << "MaxDepth=" << bestConfig.maxDepth << "\n";
    bestFile << std::fixed << std::setprecision(3)
             << "AdaBoostBeta=" << bestConfig.adaBoostBeta << "\n";
    bestFile << std::setprecision(6)
             << "Significance=" << bestSignificance << "\n";
    bestFile.close();

    std::cout << "\n" << std::string(72, '=') << std::endl;
    std::cout << "Grid scan 完成。最佳配置: " << config_tag(bestConfig) << std::endl;
    std::cout << "NTrees = " << bestConfig.nTrees
              << ", MaxDepth = " << bestConfig.maxDepth
              << ", AdaBoostBeta = " << std::fixed << std::setprecision(3) << bestConfig.adaBoostBeta
              << std::endl;
    std::cout << "最佳 significance = " << bestSignificance << std::endl;
    std::cout << "结果已写入 grid_scan_results.csv" << std::endl;
    std::cout << "最佳参数摘要已写入 grid_scan_best.txt" << std::endl;
    std::cout << std::string(72, '=') << std::endl;
}

void pTB1_plus(bool runGridScan = true) {
    TMVA::Tools::Instance();

    const char* inputFile = "/home/haoran/inputs.root";
    TFile* dataFile = TFile::Open(inputFile);
    if (!dataFile || dataFile->IsZombie()) {
        std::cerr << "错误：无法打开输入文件" << std::endl;
        return;
    }

    TTree* tree = static_cast<TTree*>(dataFile->Get("Nominal"));
    if (!tree) {
        std::cerr << "错误：无法读取 Nominal 树" << std::endl;
        dataFile->Close();
        delete dataFile;
        return;
    }

    if (runGridScan) {
        run_grid_scan(tree);
    } else {
        BDTConfig defaultConfig = {800, 4, 0.3};
        run_single_config(tree, defaultConfig);
    }

    dataFile->Close();
    delete dataFile;
}