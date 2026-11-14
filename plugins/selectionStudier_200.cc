////////////////////////////HELP////////////////////////////////
//////////////Uses new clustering algorithm to capture heavy resonance jet substructure//////////////
////////////////Last updated 17 Jul 2024 ////////////////////////////////////////////////////////////

//_top_
// system include files
#include <fastjet/JetDefinition.hh>
#include <fastjet/GhostedAreaSpec.hh>
#include <fastjet/PseudoJet.hh>
#include <fastjet/tools/Filter.hh>
#include <fastjet/ClusterSequence.hh>
//#include <fastjet/ActiveAreaSpec.hh>
#include <fastjet/ClusterSequenceArea.hh>
#include "FWCore/Framework/interface/EventSetup.h"
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <math.h>
#include "TH2.h"
#include<TRandom3.h>

#include "correction.h"
#include "ROOT/RDataFrame.hxx"
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/JetResolutionRcd.h"
#include "CondFormats/DataRecord/interface/JetResolutionScaleFactorRcd.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "PhysicsTools/CandUtils/interface/EventShapeVariables.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
// new includes
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "PhysicsTools/CandUtils/interface/Thrust.h"
//#include "Thrust.h"
#include <TTree.h>
#include <cmath>
#include "TLorentzVector.h"
#include "TVector3.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include <algorithm>   
#include "DataFormats/PatCandidates/interface/MET.h"

#include "TTree.h"
#include "TFile.h"

#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetResolutionObject.h"
#include "JetMETCorrections/Modules/interface/JetResolution.h"
#include "PhysicsTools/PatUtils/interface/SmearedJetProducerT.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>
#include "sortJets.h"
#include "BESTtoolbox.h"

#include "CacheHandler.h"
#include "BESTEvaluation.h"
#include "LHAPDF/LHAPDF.h"
#include "LHAPDF/Reweighting.h"

//#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
//#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

using namespace reco;
using namespace correction;
typedef math::XYZTLorentzVector LorentzVector;
typedef math::XYZVector Vector;
using correction::CorrectionSet;
using namespace LHAPDF;

class selectionStudier_200 : public edm::EDAnalyzer 
{
public:
   explicit selectionStudier_200(const edm::ParameterSet&);
private:
   virtual void analyze(const edm::Event&, const edm::EventSetup&);
   bool isMatchedtoSJ(std::vector<TLorentzVector> superJetTLVs, TLorentzVector candJet); 
   bool isgoodjet(const float eta, const float NHF,const float NEMF, const size_t NumConst,const float CHF,const int CHM, const float MUF, const float CEMF, bool jetPUid, const float iJet_pt);
   bool isgoodjet(const float eta, const float NHF,const float NEMF, const size_t NumConst,const float CHF,const int CHM, const float MUF, const float CEMF, int nfatjets);
   bool isHEM(const float jet_eta, const float jet_phi);
   bool fillSJVars(std::map<std::string, float> &treeVars, std::vector<fastjet::PseudoJet> iSJ, int nSuperJets);
   double top_pt_SF(double top_pt);
   std::map<std::string, std::map<std::string, std::string>> file_map;

   //init all inpaths, tokens, instrings
   edm::EDGetTokenT<std::vector<pat::Jet>> fatJetToken_;
   edm::EDGetTokenT<std::vector<pat::Jet>> jetToken_;
   edm::EDGetTokenT<std::vector<PileupSummaryInfo> > puSummaryToken_;
   edm::EDGetTokenT<edm::TriggerResults> triggerBits_;

   edm::EDGetTokenT< double > prefweight_token;


   TTree * tree;

   BESTEvaluation* BEST_;
   const CacheHandler* cache_;
   edm::FileInPath bTagSF_path;
   edm::FileInPath bTagEff_path;
   edm::EDGetTokenT<double> m_rho_token;

   edm::FileInPath path_;
   edm::FileInPath PUfile_path;
   std::string runType;
   std::string systematicType;
   std::string year;
   std::string lumiTag;

   std::string jetVetoMapName;
   edm::FileInPath jetVetoMapFile;

   std::vector<std::string> triggers;

   bool doPUID = false;
   bool doPDF = false;
   int eventnum = 0;
   int nAK4 = 0;
   int nfatjets = 0;
   int nSuperJets = 0;
   int raw_nfatjets;
   int nCategories = 3;
   int tot_nAK4_50 =0,tot_nAK4_70 = 0;
   int tot_mpp_AK4 = 0;
   int AK8_SJ_assignment[100], AK4_SJ_assignment[100];
   std::map<std::string, float> BESTmap;


   //init event variables
   bool doJEC              = true;
   bool doJER              = false;
   bool doBtagSF           = false;
   bool doPUSF             = false;
   bool doTopPtReweight    = false;
   bool doPDFWeights       = false;
   bool doPrefiringWeight  = true;
   bool _verbose           = true; // do printouts of each section for debugging
   bool debug              = false; // print out some other debug stuff
   bool runSideband        = false;
   double jet_pt[100], jet_eta[100], jet_mass[100], jet_dr[100], raw_jet_mass[100],raw_jet_pt[100],raw_jet_phi[100];
   double AK4_mass[100], AK4_E[500], leadAK8_mass[10];
   double top_pt_weight;
   double totHT = 0;
   double dijetMassOne, dijetMassTwo;
   int nfatjet_pre = 0;
   double AK4_bdisc[100], AK4_DeepJet_disc[100];
   int jet_ndaughters[100], jet_nAK4[100];
   int lab_nAK4 = 0;
   double lab_AK4_pt[100];
   double diAK8Jet_mass[100];
 

   double SJ1_BEST_scores[3],SJ2_BEST_scores[3];
   int SJ1_decision,SJ2_decision;


   double fourAK8JetMass;
   //double btag_score_uncut[100];
   double AK4_eta[100], AK4_phi[100];
   double bTag_eventWeight_T_nom, bTag_eventWeight_T_up, bTag_eventWeight_T_down, bTag_eventWeight_bc_T_up, bTag_eventWeight_bc_T_down, bTag_eventWeight_light_T_up, bTag_eventWeight_light_T_down;
   double bTag_eventWeight_M_nom, bTag_eventWeight_M_up, bTag_eventWeight_M_down, bTag_eventWeight_bc_M_up, bTag_eventWeight_bc_M_down, bTag_eventWeight_light_M_up, bTag_eventWeight_light_M_down;
   //BES variables
   int lab_AK4_AK8_parent[100];
   double jet_phi[100];
   int ntrueInt;
   double PU_eventWeight_nom;
   double deepJet_wp_loose, deepJet_wp_med, deepjet_wp_tight;
   int eventNumber;

   bool AK8_fails_veto_map[100], AK4_fails_veto_map[100];
   bool fatjet_isHEM[100], jet_pre_isHEM[100], jet_isHEM[100];
   TH2F *truebjet_eff,*truecjet_eff, *lightjet_eff;
   TH2F *truebjet_eff_med,*truecjet_eff_med, *lightjet_eff_med;

   TH2F * jetVetoMap;
   double prefiringWeight_nom;

   double bTagEffMap_PtRange, bTagEffMap_Eta_high, bTagEffMap_Eta_low;
   int bTagEffMap_nPtBins, bTagEffMap_nEtaBins;
   int SJ_nAK4_50[100],SJ_nAK4_70[100],SJ_nAK4_100[100],SJ_nAK4_125[100],SJ_nAK4_150[100],SJ_nAK4_200[100],SJ_nAK4_300[100],SJ_nAK4_400[100],SJ_nAK4_600[100],SJ_nAK4_800[100],SJ_nAK4_1000[100];
    double SJ_mass_50[100], SJ_mass_70[100],SJ_mass_100[100],superJet_mass[100],SJ_AK4_50_mass[100],SJ_AK4_70_mass[100];
   double SJ_mass_125[100], SJ_mass_150[100], SJ_mass_200[100],SJ_mass_300[100],SJ_mass_400[100],SJ_mass_600[100],SJ_mass_800[100],SJ_mass_1000[100];   

   TRandom3 *randomNum = new TRandom3(); // for JERs

   bool passesPFJet = false, passesPFHT = false;

   // variables to create a control region
   int nAK8_ET100 = 0, nAK8_ET150 = 0, nAK8_ET200 = 0, nAK8_ET250 = 0, nAK8_ET300 = 0, nAK8_ET350 = 0, nAK8_ET400 = 0;
   int nHeavyAK8_pt600_M10 = 0, nHeavyAK8_pt600_M20 = 0, nHeavyAK8_pt600_M30 = 0, nHeavyAK8_pt600_M40 = 0, nHeavyAK8_pt600_M45 = 0,  nHeavyAK8_pt600_M50 = 0,  nHeavyAK8_pt600_M55 = 0,  nHeavyAK8_pt600_M60 = 0, nHeavyAK8_pt600_M70 = 0;
   int nHeavyAK8_pt500_M10 = 0, nHeavyAK8_pt500_M20 = 0, nHeavyAK8_pt500_M30 = 0, nHeavyAK8_pt500_M40 = 0, nHeavyAK8_pt500_M45 = 0,  nHeavyAK8_pt500_M50 = 0,  nHeavyAK8_pt500_M55 = 0,  nHeavyAK8_pt500_M60 = 0, nHeavyAK8_pt500_M70 = 0;
   int nHeavyAK8_pt400_M10 = 0, nHeavyAK8_pt400_M20 = 0, nHeavyAK8_pt400_M30 = 0, nHeavyAK8_pt400_M40 = 0, nHeavyAK8_pt400_M45 = 0,  nHeavyAK8_pt400_M50 = 0,  nHeavyAK8_pt400_M55 = 0,  nHeavyAK8_pt400_M60 = 0, nHeavyAK8_pt400_M70 = 0;
   int nHeavyAK8_pt300_M10 = 0, nHeavyAK8_pt300_M20 = 0, nHeavyAK8_pt300_M30 = 0, nHeavyAK8_pt300_M40 = 0, nHeavyAK8_pt300_M45 = 0,  nHeavyAK8_pt300_M50 = 0,  nHeavyAK8_pt300_M55 = 0,  nHeavyAK8_pt300_M60 = 0, nHeavyAK8_pt300_M70 = 0;
   int nHeavyAK8_pt200_M10 = 0, nHeavyAK8_pt200_M20 = 0, nHeavyAK8_pt200_M30 = 0, nHeavyAK8_pt200_M40 = 0, nHeavyAK8_pt200_M45 = 0,  nHeavyAK8_pt200_M50 = 0,  nHeavyAK8_pt200_M55 = 0,  nHeavyAK8_pt200_M60 = 0, nHeavyAK8_pt200_M70 = 0;
   double diSuperJet_mass,diSuperJet_mass_100;;

   int nevents_200 = 0;

   //BESVars

   double AK4_m1[2], AK4_m2[2],AK4_m3[2],AK4_m4[2], AK41_E_tree[2],AK42_E_tree[2],AK43_E_tree[2],AK44_E_tree[2], AK4_m12[2],AK4_m13[2],AK4_m14[2],AK4_m23[2],AK4_m24[2],AK4_m34[2];
   double AK4_m123[2],AK4_m124[2],AK4_m134[2],AK4_m234[2], AK4_m1234[2],AK4_theta12[2],AK4_theta13[2],AK4_theta14[2],AK4_theta23[2],AK4_theta24[2],AK4_theta34[2];
   int AK41_ndaughters[2], AK41_nsubjets[2];
   double AK41_thrust[2],AK41_sphericity[2],AK41_asymmetry[2],AK41_isotropy[2],AK41_aplanarity[2],AK41_FW1[2],AK41_FW2[2],AK41_FW3[2],AK41_FW4[2];
   int AK42_ndaughters[2], AK42_nsubjets[2]; 
   double AK42_thrust[2],AK42_sphericity[2],AK42_asymmetry[2],AK42_isotropy[2],AK42_aplanarity[2],AK42_FW1[2],AK42_FW2[2],AK42_FW3[2],AK42_FW4[2];
   int AK43_ndaughters[2], AK43_nsubjets[2]; 
   double AK43_thrust[2],AK43_sphericity[2],AK43_asymmetry[2],AK43_isotropy[2],AK43_aplanarity[2],AK43_FW1[2],AK43_FW2[2],AK43_FW3[2],AK43_FW4[2];
   double SJ_thrust[2],SJ_sphericity[2],SJ_asymmetry[2],SJ_isotropy[2],SJ_aplanarity[2],SJ_FW1[2],SJ_FW2[2],SJ_FW3[2],SJ_FW4[2];

   ///////////////////////////////////////////////

   // btag scale factor stuff
   std::unique_ptr<CorrectionSet> cset;
   Correction::Ref cset_corrector_bc;
   Correction::Ref cset_corrector_light; 

   // Jet correction uncertainty classes
   JetCorrectionUncertainty *jecUnc_AK4;
   JetCorrectionUncertainty *jecUnc_AK8;
   std::unique_ptr<CorrectionSet> PUjson;
   Correction::Ref PUjson_year;

   // jet veto map stuff
   double jetVetoMap_XRange, jetVetoMap_YRange, jetVetoMap_Xmin, jetVetoMap_Ymin;
   int jetVetoMap_nBinsX, jetVetoMap_nBinsY;

};
bool selectionStudier_200::isMatchedtoSJ(std::vector<TLorentzVector> superJetTLVs, TLorentzVector candJet)
{
   for(auto iJet = superJetTLVs.begin(); iJet!=superJetTLVs.end(); iJet++)
   {
      if( abs(candJet.Angle(iJet->Vect())) < 0.001) return true;  
   }
   return false;
}

//_constructor_
selectionStudier_200::selectionStudier_200(const edm::ParameterSet& iConfig):
   path_ (iConfig.getParameter<edm::FileInPath>("BESTpath"))
   
{
   // import parameters from cfg, set variables needed to analyze the event
   runType        = iConfig.getParameter<std::string>("runType");
   systematicType = iConfig.getParameter<std::string>("systematicType");
   year           = iConfig.getParameter<std::string>("year");
   std::cout << "path: " << path_.fullPath() << std::endl;
   cache_ = new CacheHandler(path_);
   BEST_  = new BESTEvaluation(cache_);
   BEST_->configure(iConfig);
   _verbose           = iConfig.getParameter<bool>("verbose");
   runSideband        = iConfig.getParameter<bool>("runSideband");

   jetVetoMapName = iConfig.getParameter<std::string>("jetVetoMapName");
   jetVetoMapFile = iConfig.getParameter<edm::FileInPath>("jetVetoMapFile");


   prefweight_token     = consumes< double >(edm::InputTag("prefiringweight:nonPrefiringProb"));

   doPDF = iConfig.getParameter<bool>("doPDF");

   triggerBits_ = consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("bits"));

   if( (runType.find("MC") != std::string::npos) || (runType.find("Suu") != std::string::npos))    //don't want these variables for data
   {
      // these will only be done for MC
      doBtagSF = true;
      doJER    = true;
      doPUSF   = true;
      if(doPDF)doPDFWeights = true; 
      else {doPDFWeights = false;}

      if ((runType.find("TTTo") != std::string::npos) || (runType.find("TTJets") != std::string::npos)) doTopPtReweight = true;  

      bTagSF_path    = iConfig.getParameter<edm::FileInPath>("bTagSF_path");
      bTagEff_path   = iConfig.getParameter<edm::FileInPath>("bTagEff_path");

   }
   PUfile_path        = iConfig.getParameter<edm::FileInPath>("PUfile_path");

   // prefiring weights
   prefweight_token     = consumes< double >(edm::InputTag("prefiringweight:nonPrefiringProb"));

   doPUID = iConfig.getParameter<bool>("doPUID");

   m_rho_token  = consumes<double>(edm::InputTag("fixedGridRhoAll", "", "RECO"));

   fatJetToken_ = consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("fatJetCollection"));
   jetToken_    = consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("jetCollection"));

   edm::Service<TFileService> fs;      
   tree = fs->make<TTree>(  ("tree_"+systematicType).c_str(), ("tree_"+systematicType).c_str());
   TFile *bTagEff_file;
   TFile *jetVetoMap_file;

   if(year == "2018")
   {
      //deepJet_wp_loose = 0.0490;
      deepJet_wp_med   = 0.2783;
      deepjet_wp_tight = 0.7100;
      lumiTag = "Collisions18_UltraLegacy_goldenJSON";
      triggers = {"HLT_PFJet500_v", "HLT_PFHT1050_v"};
   }
   else if(year == "2017")
   {
      //deepJet_wp_loose = 0.0532;
      deepJet_wp_med   = 0.3040;
      deepjet_wp_tight = 0.7476;
      lumiTag = "Collisions17_UltraLegacy_goldenJSON";
      triggers = {"HLT_PFJet500_v", "HLT_PFHT1050_v"};
   }
   else if(year == "2016")
   {
      //deepJet_wp_loose = 0.0480;
      deepJet_wp_med   = 0.2489;
      deepjet_wp_tight = 0.6377;
      lumiTag = "Collisions16_UltraLegacy_goldenJSON";
      triggers  = {"HLT_PFHT900_v", "HLT_PFJet450_v"};
   }
   else if(year == "2015")
   {
      //deepJet_wp_loose = 0.0508;
      deepJet_wp_med   = 0.2598;
      deepjet_wp_tight = 0.6502;
      lumiTag = "Collisions16_UltraLegacy_goldenJSON";
      triggers = {"HLT_PFHT900_v", "HLT_PFJet450_v"};
   }
   else
   {
      std::cout << "Incorrect year: " << year << std::endl;
      return; // return cut
   }


   ///////////////////////////////////////////////////////////////
   //////////////////////// B-tagging stuff //////////////////////
   ///////////////////////////////////////////////////////////////

   if( (runType.find("MC") != std::string::npos) || (runType.find("Suu") != std::string::npos))    //don't want these variables for data
   {

      if(_verbose)std::cout << "Setting up b-tag efficiency maps" << std::endl;
      bTagEff_file = new TFile( bTagEff_path.fullPath().c_str() );   //this path might need to be updated

      truebjet_eff = (TH2F*)bTagEff_file->Get("h_effbJets_tight");   
      truecjet_eff = (TH2F*)bTagEff_file->Get("h_effcJets_tight");  
      lightjet_eff = (TH2F*)bTagEff_file->Get("h_effLightJets_tight"); 

      truebjet_eff_med = (TH2F*)bTagEff_file->Get("h_effbJets_med"); 
      truecjet_eff_med = (TH2F*)bTagEff_file->Get("h_effcJets_med"); 
      lightjet_eff_med = (TH2F*)bTagEff_file->Get("h_effLightJets_med"); 


      bTagEffMap_PtRange = lightjet_eff->GetXaxis()->GetXmax();
      bTagEffMap_Eta_high = lightjet_eff->GetYaxis()->GetXmax();
      bTagEffMap_Eta_low  = lightjet_eff->GetYaxis()->GetXmin();

      bTagEffMap_nPtBins = lightjet_eff->GetNbinsX();
      bTagEffMap_nEtaBins = lightjet_eff->GetNbinsY();

      if(_verbose)std::cout << "Setting up b-tag correctionlib corrector" << std::endl;

      cset = CorrectionSet::from_file(bTagSF_path.fullPath());
      cset_corrector_bc    = cset->at("deepJet_mujets");   //deepJet_comb,    // deepJet_mujets -> what we were using originally, deepJet_ttbar -> what BTV suggested we could use, UPDATE: deepJet_ttbar is not recognized!!!
      if(_verbose)std::cout << "Got bc b-tag correctionlib corrector" << std::endl;
      cset_corrector_light = cset->at("deepJet_incl");   //deepJet_incl
      

      //get corrections from PU file
      PUjson = CorrectionSet::from_file(PUfile_path.fullPath());
      PUjson_year = PUjson->at(lumiTag);
      if(_verbose)std::cout << "Set up pileup correctionlib corrector" << std::endl;
   }

   if(_verbose)std::cout << "Setting up jet veto map files and ranges" << std::endl;


   //////////////////////////////////////
   //////// Jet veto map stuff //////////
   ////////////////////////////////////// 

   jetVetoMap_file = new TFile( jetVetoMapFile.fullPath().c_str() ); 
   jetVetoMap = (TH2F*)jetVetoMap_file->Get(jetVetoMapName.c_str());
   jetVetoMap_XRange = jetVetoMap->GetXaxis()->GetXmax() - jetVetoMap->GetXaxis()->GetXmin();
   jetVetoMap_YRange = jetVetoMap->GetYaxis()->GetXmax() - jetVetoMap->GetYaxis()->GetXmin();

   jetVetoMap_Xmin = jetVetoMap->GetXaxis()->GetXmin();
   jetVetoMap_Ymin = jetVetoMap->GetYaxis()->GetXmin();
   jetVetoMap_nBinsX = jetVetoMap->GetNbinsX();
   jetVetoMap_nBinsY = jetVetoMap->GetNbinsY();


   if(_verbose)std::cout << "Initializing TTree variables." << std::endl;

   ///////////////////////////////////////
   ////////// init tree branches /////////
   ///////////////////////////////////////

   tree->Branch("nfatjets", &nfatjets, "nfatjets/I");
   tree->Branch("nSuperJets", &nSuperJets, "nSuperJets/I");
   tree->Branch("passesPFHT", &passesPFHT  , "passesPFHT/O");
   tree->Branch("passesPFJet", &passesPFJet  , "passesPFJet/O");
   tree->Branch("SJ_mass_50", SJ_mass_50, "SJ_mass_50[nSuperJets]/D");
   tree->Branch("SJ_mass_100", SJ_mass_100, "SJ_mass_100[nSuperJets]/D");


   tree->Branch("nAK4", &nAK4, "nAK4/I");
   tree->Branch("AK4_DeepJet_disc", AK4_DeepJet_disc, "AK4_DeepJet_disc[nAK4]/D");

   tree->Branch("nfatjet_pre",&nfatjet_pre, "nfatjet_pre/I");
   tree->Branch("totHT",&totHT, "totHT/D");
   
   tree->Branch("jet_pt", jet_pt, "jet_pt[nfatjets]/D");
   tree->Branch("jet_eta", jet_eta, "jet_eta[nfatjets]/D");
   tree->Branch("jet_phi", jet_phi, "jet_phi[nfatjets]/D");
   tree->Branch("AK8_fails_veto_map", AK8_fails_veto_map, "AK8_fails_veto_map[nfatjets]/O");

   tree->Branch("jet_isHEM", jet_isHEM, "jet_isHEM[nfatjets]/O");
   tree->Branch("fatjet_isHEM", fatjet_isHEM, "fatjet_isHEM[nfatjets]/O");
   tree->Branch("jet_pre_isHEM", jet_pre_isHEM, "jet_pre_isHEM[nfatjet_pre]/O");

   tree->Branch("lab_nAK4", &lab_nAK4, "lab_nAK4/I");
   tree->Branch("lab_AK4_pt", lab_AK4_pt, "lab_AK4_pt[lab_nAK4]/D");
   tree->Branch("AK4_eta", AK4_eta  , "AK4_eta[lab_nAK4]/D");
   tree->Branch("AK4_phi", AK4_phi  , "AK4_phi[lab_nAK4]/D");
   tree->Branch("AK4_fails_veto_map", AK4_fails_veto_map  , "AK4_fails_veto_map[lab_nAK4]/O");

   tree->Branch("jet_mass", jet_mass, "jet_mass[nfatjets]/D");
   tree->Branch("AK4_mass", AK4_mass, "AK4_mass[nAK4]/D");

   tree->Branch("dijetMassOne", &dijetMassOne, "dijetMassOne/D");
   tree->Branch("dijetMassTwo", &dijetMassTwo, "dijetMassTwo/D");

   tree->Branch("prefiringWeight_nom",  &prefiringWeight_nom, "prefiringWeight_nom/D");

   tree->Branch("SJ_nAK4_50", SJ_nAK4_50, "SJ_nAK4_50[nSuperJets]/I");
   tree->Branch("SJ_nAK4_70", SJ_nAK4_70, "SJ_nAK4_70[nSuperJets]/I");
   tree->Branch("SJ_nAK4_100", SJ_nAK4_100, "SJ_nAK4_100[nSuperJets]/I");
   tree->Branch("SJ_nAK4_125", SJ_nAK4_125, "SJ_nAK4_125[nSuperJets]/I");
   tree->Branch("SJ_nAK4_150", SJ_nAK4_150, "SJ_nAK4_150[nSuperJets]/I");
   tree->Branch("SJ_nAK4_200", SJ_nAK4_200, "SJ_nAK4_200[nSuperJets]/I");
   tree->Branch("SJ_nAK4_300", SJ_nAK4_300, "SJ_nAK4_300[nSuperJets]/I");
   tree->Branch("SJ_nAK4_400", SJ_nAK4_400, "SJ_nAK4_400[nSuperJets]/I");
   
   tree->Branch("nCategories", &nCategories, "nCategories/I");
   tree->Branch("SJ1_BEST_scores", SJ1_BEST_scores, "SJ1_BEST_scores[nCategories]/D");
   tree->Branch("SJ2_BEST_scores", SJ2_BEST_scores, "SJ2_BEST_scores[nCategories]/D");
   tree->Branch("SJ1_decision", &SJ1_decision, "SJ1_decision/I");
   tree->Branch("SJ2_decision", &SJ2_decision, "SJ2_decision/I");

   /////////// (2) Add grid variables here ///////////
   tree->Branch("nAK8_ET100",  &nAK8_ET100, "nAK8_ET100/I");
   tree->Branch("nAK8_ET150",  &nAK8_ET150, "nAK8_ET150/I");
   tree->Branch("nAK8_ET200",  &nAK8_ET200, "nAK8_ET200/I");
   tree->Branch("nAK8_ET250",  &nAK8_ET250, "nAK8_ET250/I");
   tree->Branch("nAK8_ET300",  &nAK8_ET300, "nAK8_ET300/I");
   tree->Branch("nAK8_ET350",  &nAK8_ET350, "nAK8_ET350/I");
   tree->Branch("nAK8_ET400",  &nAK8_ET400, "nAK8_ET400/I");
   tree->Branch("nHeavyAK8_pt600_M10",  &nHeavyAK8_pt600_M10, "nHeavyAK8_pt600_M10/I");
   tree->Branch("nHeavyAK8_pt600_M20",  &nHeavyAK8_pt600_M20, "nHeavyAK8_pt600_M20/I");
   tree->Branch("nHeavyAK8_pt600_M30",  &nHeavyAK8_pt600_M30, "nHeavyAK8_pt600_M30/I");
   tree->Branch("nHeavyAK8_pt600_M40",  &nHeavyAK8_pt600_M40, "nHeavyAK8_pt600_M40/I");
   tree->Branch("nHeavyAK8_pt600_M45",  &nHeavyAK8_pt600_M45, "nHeavyAK8_pt600_M45/I");
   tree->Branch("nHeavyAK8_pt600_M50",  &nHeavyAK8_pt600_M50, "nHeavyAK8_pt600_M50/I");
   tree->Branch("nHeavyAK8_pt600_M55",  &nHeavyAK8_pt600_M55, "nHeavyAK8_pt600_M55/I");
   tree->Branch("nHeavyAK8_pt600_M60",  &nHeavyAK8_pt600_M60, "nHeavyAK8_pt600_M60/I");
   tree->Branch("nHeavyAK8_pt600_M70",  &nHeavyAK8_pt600_M70, "nHeavyAK8_pt600_M70/I");

   tree->Branch("nHeavyAK8_pt500_M10",  &nHeavyAK8_pt500_M10, "nHeavyAK8_pt500_M10/I");
   tree->Branch("nHeavyAK8_pt500_M20",  &nHeavyAK8_pt500_M20, "nHeavyAK8_pt500_M20/I");
   tree->Branch("nHeavyAK8_pt500_M30",  &nHeavyAK8_pt500_M30, "nHeavyAK8_pt500_M30/I");
   tree->Branch("nHeavyAK8_pt500_M40",  &nHeavyAK8_pt500_M40, "nHeavyAK8_pt500_M40/I");
   tree->Branch("nHeavyAK8_pt500_M45",  &nHeavyAK8_pt500_M45, "nHeavyAK8_pt500_M45/I");
   tree->Branch("nHeavyAK8_pt500_M50",  &nHeavyAK8_pt500_M50, "nHeavyAK8_pt500_M50/I");
   tree->Branch("nHeavyAK8_pt500_M55",  &nHeavyAK8_pt500_M55, "nHeavyAK8_pt500_M55/I");
   tree->Branch("nHeavyAK8_pt500_M60",  &nHeavyAK8_pt500_M60, "nHeavyAK8_pt500_M60/I");
   tree->Branch("nHeavyAK8_pt500_M70",  &nHeavyAK8_pt500_M70, "nHeavyAK8_pt500_M70/I");


   tree->Branch("nHeavyAK8_pt400_M10",  &nHeavyAK8_pt400_M10, "nHeavyAK8_pt400_M10/I");
   tree->Branch("nHeavyAK8_pt400_M20",  &nHeavyAK8_pt400_M20, "nHeavyAK8_pt400_M20/I");
   tree->Branch("nHeavyAK8_pt400_M30",  &nHeavyAK8_pt400_M30, "nHeavyAK8_pt400_M30/I");
   tree->Branch("nHeavyAK8_pt400_M40",  &nHeavyAK8_pt400_M40, "nHeavyAK8_pt400_M40/I");
   tree->Branch("nHeavyAK8_pt400_M45",  &nHeavyAK8_pt400_M45, "nHeavyAK8_pt400_M45/I");
   tree->Branch("nHeavyAK8_pt400_M50",  &nHeavyAK8_pt400_M50, "nHeavyAK8_pt400_M50/I");
   tree->Branch("nHeavyAK8_pt400_M55",  &nHeavyAK8_pt400_M55, "nHeavyAK8_pt400_M55/I");
   tree->Branch("nHeavyAK8_pt400_M60",  &nHeavyAK8_pt400_M60, "nHeavyAK8_pt400_M60/I");
   tree->Branch("nHeavyAK8_pt400_M70",  &nHeavyAK8_pt400_M70, "nHeavyAK8_pt400_M70/I");

   tree->Branch("nHeavyAK8_pt300_M10",  &nHeavyAK8_pt300_M10, "nHeavyAK8_pt300_M10/I");
   tree->Branch("nHeavyAK8_pt300_M20",  &nHeavyAK8_pt300_M20, "nHeavyAK8_pt300_M20/I");
   tree->Branch("nHeavyAK8_pt300_M30",  &nHeavyAK8_pt300_M30, "nHeavyAK8_pt300_M30/I");
   tree->Branch("nHeavyAK8_pt300_M40",  &nHeavyAK8_pt300_M40, "nHeavyAK8_pt300_M40/I");
   tree->Branch("nHeavyAK8_pt300_M45",  &nHeavyAK8_pt300_M45, "nHeavyAK8_pt300_M45/I");
   tree->Branch("nHeavyAK8_pt300_M50",  &nHeavyAK8_pt300_M50, "nHeavyAK8_pt300_M50/I");
   tree->Branch("nHeavyAK8_pt300_M55",  &nHeavyAK8_pt300_M55, "nHeavyAK8_pt300_M55/I");
   tree->Branch("nHeavyAK8_pt300_M60",  &nHeavyAK8_pt300_M60, "nHeavyAK8_pt300_M60/I");
   tree->Branch("nHeavyAK8_pt300_M70",  &nHeavyAK8_pt300_M70, "nHeavyAK8_pt300_M70/I");

   tree->Branch("nHeavyAK8_pt200_M10",  &nHeavyAK8_pt200_M10, "nHeavyAK8_pt200_M10/I");
   tree->Branch("nHeavyAK8_pt200_M20",  &nHeavyAK8_pt200_M20, "nHeavyAK8_pt200_M20/I");
   tree->Branch("nHeavyAK8_pt200_M30",  &nHeavyAK8_pt200_M30, "nHeavyAK8_pt200_M30/I");
   tree->Branch("nHeavyAK8_pt200_M40",  &nHeavyAK8_pt200_M40, "nHeavyAK8_pt200_M40/I");
   tree->Branch("nHeavyAK8_pt200_M45",  &nHeavyAK8_pt200_M45, "nHeavyAK8_pt200_M45/I");
   tree->Branch("nHeavyAK8_pt200_M50",  &nHeavyAK8_pt200_M50, "nHeavyAK8_pt200_M50/I");
   tree->Branch("nHeavyAK8_pt200_M55",  &nHeavyAK8_pt200_M55, "nHeavyAK8_pt200_M55/I");
   tree->Branch("nHeavyAK8_pt200_M60",  &nHeavyAK8_pt200_M60, "nHeavyAK8_pt200_M60/I");
   tree->Branch("nHeavyAK8_pt200_M70",  &nHeavyAK8_pt200_M70, "nHeavyAK8_pt200_M70/I");
   tree->Branch("diSuperJet_mass",  &diSuperJet_mass, "diSuperJet_mass/D");
    tree->Branch("diSuperJet_mass_100",&diSuperJet_mass_100, "diSuperJet_mass_100/D");
   
   tree->Branch("superJet_mass", superJet_mass, "superJet_mass[nSuperJets]/D");
    tree->Branch("eventNumber", &eventNumber  , "eventNumber/I");

  


   ///////////////////////////////////////////////

   if( (runType.find("MC") != std::string::npos) || (runType.find("Suu") != std::string::npos))    //don't want these variables for data
   {
      if(_verbose)std::cout << "Setting up MC-specific stuff (genParts tokens, MC-specific TTree vars)" << std::endl;

      puSummaryToken_         = consumes<std::vector<PileupSummaryInfo>>(iConfig.getParameter<edm::InputTag>("pileupCollection"));

      if(doTopPtReweight) tree->Branch("top_pt_weight", &top_pt_weight, "top_pt_weight/D");

      tree->Branch("bTag_eventWeight_T_nom", &bTag_eventWeight_T_nom  , "bTag_eventWeight_T_nom/D");  /// tight WP event weight
      tree->Branch("bTag_eventWeight_M_nom", &bTag_eventWeight_M_nom  , "bTag_eventWeight_M_nom/D");  /// med WP event weight

      if(doPUSF)  tree->Branch("PU_eventWeight_nom", &PU_eventWeight_nom, "PU_eventWeight_nom/D");


   }
   else if(runType == "Data")
   {
         std::cout <<"Running as data ..." << std::endl;
   }

}





// returns bool if jet (or more generally, object) is within the HEM region
bool selectionStudier_200::isHEM(const float jet_eta, const float jet_phi)
{
   if(year != "2018") return false; // HEM is only relevant for 2018

   if( (jet_phi >  -1.57)&&( jet_phi < -0.87) )
   {
      if( (jet_eta > -3.0)&&(jet_eta < -1.3))return true;

   }
   return false;
}


// bool corresponding to if AK4 jet passes tight ID
bool selectionStudier_200::isgoodjet(const float eta, const float NHF,const float NEMF, const size_t NumConst,const float CHF,const int CHM, const float MUF, const float CEMF,bool jetPUid, const float iJet_pt)
{
   if( (abs(eta) > 2.4)) return false;

   // only apply to AK4 jets
   // apply the MEDIUM PU jet id https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL
   if( (!jetPUid) && (iJet_pt < 50.0)) return false;

   if ((NHF>0.9) || (NEMF>0.9) || (NumConst<1) || (CHF<0.) || (CHM<0) || (MUF > 0.8) || (CEMF > 0.8)) 
   {
         return false;
   }
   else
   {
      return true;
   }

}
// checks AK8 jet (tight) ID
bool selectionStudier_200::isgoodjet(const float eta, const float NHF,const float NEMF, const size_t NumConst,const float CHF,const int CHM, const float MUF, const float CEMF, int nfatjets)
{
   if ( (nfatjets < 2) && (abs(eta) > 2.4) ) return false;
   else if ( (nfatjets >= 2) && (abs(eta) > 1.4) ) return false;

   if ((NHF>0.9) || (NEMF>0.9) || (NumConst<1) || (CHF<0.) || (CHM<0) || (MUF > 0.8) || (CEMF > 0.8)) 
      {
         return false;
      }
   else{ return true;}

}


// returns the top pt scale factor as detailed here - https://twiki.cern.ch/twiki/bin/view/CMS/TopPtReweighting#Run_1_strategy_Obsolete
double selectionStudier_200::top_pt_SF(double top_pt)
{
   if (top_pt > 500.) top_pt = 500.;
   //$SF(p_T)=e^{0.0615-0.0005\cdot p_T}$ for data/POWHEG+Pythia8
   //return 0.103*exp(-0.0118*top_pt) -0.000134*top_pt+ 0.973;
   return exp(0.0615-0.0005*top_pt);  // this is the scale factor based on data aka data-NLO and data-NNLO weights
}

bool selectionStudier_200::fillSJVars(std::map<std::string, float> &treeVars, std::vector<fastjet::PseudoJet> iSJ, int nSuperJets )
{
   bool testNewVars = false; // include new SJ vars to give to NN

   double superJetpx=0,superJetpy=0,superJetpz=0,superJetE=0;
   for(auto iP = iSJ.begin();iP != iSJ.end();iP++)
   {
      superJetpx+=iP->px();superJetpy+=iP->py();superJetpz+=iP->pz();superJetE +=iP->E();
   }

   TLorentzVector superJetTLV(superJetpx,superJetpy,superJetpz,superJetE);    //Lorentz vector representing jet axis -> now minimize the parallel momentum
      
   //boost COM
   std::vector<fastjet::PseudoJet> boostedSuperJetPart;

   //boost particles in SuperJet to COM frame
   for(auto iP = iSJ.begin();iP != iSJ.end();iP++)
   {
      TLorentzVector iP_(iP->px(),iP->py(),iP->pz(),iP->E());
      iP_.Boost(-superJetTLV.Px()/superJetTLV.E(),-superJetTLV.Py()/superJetTLV.E(),-superJetTLV.Pz()/superJetTLV.E());
      boostedSuperJetPart.push_back(fastjet::PseudoJet(iP_.Px(),iP_.Py(),iP_.Pz(),iP_.E()));
   }

   ///////////get a bunch of versions of SJ particle collection to calclate BES variables
   std::vector<TLorentzVector> boostedSuperJetPart_TLV;
   std::vector<reco::LeafCandidate> boostedSuperJetPart_LC;
   std::vector<math::XYZVector> boostedSuperJetPart_XYZ;
   double sumPz = 0, sumP = 0;
   for(auto iP_= boostedSuperJetPart.begin(); iP_ !=boostedSuperJetPart.end(); iP_++)
   {   
      boostedSuperJetPart_TLV.push_back(TLorentzVector(iP_->px(),iP_->py(),iP_->pz(),iP_->E()));
      boostedSuperJetPart_LC.push_back(reco::LeafCandidate(+1, reco::Candidate::LorentzVector(iP_->px(),iP_->py(),iP_->pz(),iP_->E())));
      boostedSuperJetPart_XYZ.push_back(math::XYZVector( iP_->px(),iP_->py(),iP_->pz() ));
      sumPz += abs(iP_->pz());
      sumP += abs(sqrt(pow(iP_->pz(),2) + pow(iP_->px(),2)+ pow(iP_->py(),2)));
   }

   ///reclustering SuperJet that is now boosted into the SJ COM frame
   double R = 0.4;

   //fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, R);
   fastjet::JetDefinition jet_def(fastjet::cambridge_algorithm, R);
   fastjet::ClusterSequence cs_jet(boostedSuperJetPart, jet_def); 
   std::vector<fastjet::PseudoJet> jetsFJ_jet = fastjet::sorted_by_E(cs_jet.inclusive_jets(0.0));

   double SJ_25_px = 0, SJ_25_py=0,SJ_25_pz=0,SJ_25_E=0;
   double SJ_50_px = 0, SJ_50_py=0,SJ_50_pz=0,SJ_50_E=0;
   double SJ_75_px = 0, SJ_75_py=0,SJ_75_pz=0,SJ_75_E=0;
   double SJ_100_px = 0, SJ_100_py=0,SJ_100_pz=0,SJ_100_E=0;
   double SJ_150_px = 0, SJ_150_py=0,SJ_150_pz=0,SJ_150_E=0;
   double SJ_200_px = 0, SJ_200_py=0,SJ_200_pz=0,SJ_200_E=0;
   double SJ_300_px = 0, SJ_300_py=0,SJ_300_pz=0,SJ_300_E=0;
   double SJ_400_px = 0, SJ_400_py=0,SJ_400_pz=0,SJ_400_E=0;
   double SJ_500_px = 0, SJ_500_py=0,SJ_500_pz=0,SJ_500_E=0;
   double SJ_800_px = 0, SJ_800_py=0,SJ_800_pz=0,SJ_800_E=0;
   double SJ_1000_px = 0, SJ_1000_py=0,SJ_1000_pz=0,SJ_1000_E=0;

   int SJ_nAK4_1 = 0, SJ_nAK4_5 = 0, SJ_nAK4_10_ = 0;
   int SJ_nAK4_25_ = 0;
   int SJ_nAK4_50_ = 0, SJ_nAK4_75_ = 0, SJ_nAK4_100_ = 0,SJ_nAK4_150_ = 0,SJ_nAK4_200_ = 0,SJ_nAK4_300_ = 0;
   int SJ_nAK4_400_ = 0,SJ_nAK4_500_ = 0,SJ_nAK4_800_ = 0,SJ_nAK4_1000_ = 0;


   std::vector<TLorentzVector> AK41_parts;
   std::vector<TLorentzVector> AK42_parts;
   std::vector<TLorentzVector> AK43_parts;
   std::vector<TLorentzVector> AK44_parts;

   double AK41_px = 0, AK41_py=0,AK41_pz = 0, AK41_E = 0;
   double AK42_px = 0, AK42_py=0,AK42_pz = 0, AK42_E = 0;
   double AK43_px = 0, AK43_py=0,AK43_pz = 0, AK43_E = 0;
   double AK44_px = 0, AK44_py=0,AK44_pz = 0, AK44_E = 0;


   
   if (jetsFJ_jet.size() < 4) // fewer than 4 reclustered CA4 jets in superjet, this shouldn't reasonably happen ...
   {
      return false; // RETURN CUT
      std::cout << "A pseudojet vector has a size smaller than 2 - not reclustering enough jets from pool of particles - " << jetsFJ_jet.size() << std::endl;
   }

   int pseudoJetNum = 0;
   for (auto iPJ=jetsFJ_jet.begin(); iPJ<jetsFJ_jet.end(); iPJ++)                             
   {

      // do calculations of AK4 btagged particle ratios for leading 4 AK4 jets
      std::vector<fastjet::PseudoJet> iPJ_daughters = iPJ->constituents();
            
      if(pseudoJetNum == 0)
      {   
         treeVars["AK41_px"] = iPJ->px();
         treeVars["AK41_py"] = iPJ->py();
         treeVars["AK41_pz"] = iPJ->pz();
         treeVars["AK41_E"] = iPJ->E();
         for(auto iPart = iPJ_daughters.begin(); iPart != iPJ_daughters.end(); iPart++)
         {
            AK41_parts.push_back(TLorentzVector(iPart->px(),iPart->py(),iPart->pz(),iPart->E()));
            AK41_px+=iPart->px();AK41_py+=iPart->py();AK41_pz+=iPart->pz();AK41_E+=iPart->E();
         }

      }
      else if(pseudoJetNum == 1)
      {
         treeVars["AK42_px"] = iPJ->px();
         treeVars["AK42_py"] = iPJ->py();
         treeVars["AK42_pz"] = iPJ->pz();
         treeVars["AK42_E"] = iPJ->E();
         for(auto iPart = iPJ_daughters.begin(); iPart != iPJ_daughters.end(); iPart++)
         {
            AK42_parts.push_back(TLorentzVector(iPart->px(),iPart->py(),iPart->pz(),iPart->E()));
            AK42_px+=iPart->px();AK42_py+=iPart->py();AK42_pz+=iPart->pz();AK42_E+=iPart->E();
         }
      }
      else if(pseudoJetNum == 2)
      {
         treeVars["AK43_px"] = iPJ->px();
         treeVars["AK43_py"] = iPJ->py();
         treeVars["AK43_pz"] = iPJ->pz();
         treeVars["AK43_E"] = iPJ->E();
         for(auto iPart = iPJ_daughters.begin(); iPart != iPJ_daughters.end(); iPart++)
         {
            AK43_parts.push_back(TLorentzVector(iPart->px(),iPart->py(),iPart->pz(),iPart->E()));
            AK43_px+=iPart->px();AK43_py+=iPart->py();AK43_pz+=iPart->pz();AK43_E+=iPart->E();
         }
      }
      else if(pseudoJetNum == 3)
      {
         treeVars["AK44_px"] = iPJ->px();
         treeVars["AK44_py"] = iPJ->py();
         treeVars["AK44_pz"] = iPJ->pz();
         treeVars["AK44_E"] = iPJ->E();      
         for(auto iPart = iPJ_daughters.begin(); iPart != iPJ_daughters.end(); iPart++)
         {
            AK44_parts.push_back(TLorentzVector(iPart->px(),iPart->py(),iPart->pz(),iPart->E()));
            AK44_px+=iPart->px();AK44_py+=iPart->py();AK44_pz+=iPart->pz();AK44_E+=iPart->E();
         }
      }

      if(iPJ->E()>1.)  SJ_nAK4_1++;   // this is just used for a fraction calculation
      if(iPJ->E()>5.)  SJ_nAK4_5++;   // this is just used for a fraction calculation
      if(iPJ->E()>10.) SJ_nAK4_10_++;

      if(iPJ->E()>25.)
      {
         SJ_25_px+=iPJ->px();SJ_25_py+=iPJ->py();SJ_25_pz+=iPJ->pz();SJ_25_E+=iPJ->E();
         SJ_nAK4_25_++;
      }

      if(iPJ->E()>50.)
      {
         SJ_50_px+=iPJ->px();SJ_50_py+=iPJ->py();SJ_50_pz+=iPJ->pz();SJ_50_E+=iPJ->E();
         SJ_nAK4_50_++;
      }
      if(iPJ->E()>75.)
      {
         SJ_75_px+=iPJ->px();SJ_75_py+=iPJ->py();SJ_75_pz+=iPJ->pz();SJ_75_E+=iPJ->E();
         SJ_nAK4_75_++;
      }
      if(iPJ->E()>100)
      {
         SJ_100_px+=iPJ->px();SJ_100_py+=iPJ->py();SJ_100_pz+=iPJ->pz();SJ_100_E+=iPJ->E();
         SJ_nAK4_100_++; 
      }

      if(iPJ->E()>150)
      {
         SJ_150_px+=iPJ->px();SJ_150_py+=iPJ->py();SJ_150_pz+=iPJ->pz();SJ_150_E+=iPJ->E();
         SJ_nAK4_150_++; 
      }

      if(iPJ->E()>200)
      {
         SJ_200_px+=iPJ->px();SJ_200_py+=iPJ->py();SJ_200_pz+=iPJ->pz();SJ_200_E+=iPJ->E();
         SJ_nAK4_200_++; 
      }
      if(iPJ->E()>300)
      {
         SJ_300_px+=iPJ->px();SJ_300_py+=iPJ->py();SJ_300_pz+=iPJ->pz();SJ_300_E+=iPJ->E();
         SJ_nAK4_300_++; 
      }
      if(iPJ->E()>400)
      {
         SJ_400_px+=iPJ->px();SJ_400_py+=iPJ->py();SJ_400_pz+=iPJ->pz();SJ_400_E+=iPJ->E();
         SJ_nAK4_400_++; 
      }
      if(iPJ->E()>500)
      {
         SJ_500_px+=iPJ->px();SJ_500_py+=iPJ->py();SJ_500_pz+=iPJ->pz();SJ_500_E+=iPJ->E();
         SJ_nAK4_500_++; 
      }
      if(iPJ->E()>800)
      {
         SJ_800_px+=iPJ->px();SJ_800_py+=iPJ->py();SJ_800_pz+=iPJ->pz();SJ_800_E+=iPJ->E();
         SJ_nAK4_800_++; 
      }
      if(iPJ->E()>1000)
      {
         SJ_1000_px+=iPJ->px();SJ_1000_py+=iPJ->py();SJ_1000_pz+=iPJ->pz();SJ_1000_E+=iPJ->E();
         SJ_nAK4_1000_++; 
      }
      pseudoJetNum++;
   }

   boostedSuperJetPart.clear();   //shouldn't be needed, just in case

   /////annoying process of getting the BES information for the reclustered AK4 jets

   TVector3 AK41_boost(AK41_px/AK41_E,AK41_py/AK41_E,AK41_pz/AK41_E);
   TVector3 AK42_boost(AK42_px/AK42_E,AK42_py/AK42_E,AK42_pz/AK42_E);
   TVector3 AK43_boost(AK43_px/AK43_E,AK43_py/AK43_E,AK43_pz/AK43_E);
   TVector3 AK44_boost(AK44_px/AK44_E,AK44_py/AK44_E,AK44_pz/AK44_E);

   std::vector<TLorentzVector> boostedAK41_Part_TLV;
   std::vector<reco::LeafCandidate>  boostedAK41_Part_LC;
   std::vector<math::XYZVector>  boostedAK41_Part_XYZ;

   std::vector<TLorentzVector> boostedAK42_Part_TLV;
   std::vector<reco::LeafCandidate>  boostedAK42_Part_LC;
   std::vector<math::XYZVector>  boostedAK42_Part_XYZ;

   std::vector<TLorentzVector> boostedAK43_Part_TLV;
   std::vector<reco::LeafCandidate>  boostedAK43_Part_LC;
   std::vector<math::XYZVector>  boostedAK43_Part_XYZ;

   std::vector<TLorentzVector> boostedAK44_Part_TLV;
   std::vector<reco::LeafCandidate>  boostedAK44_Part_LC;
   std::vector<math::XYZVector>  boostedAK44_Part_XYZ;

   double sumPz_AK41 =0,sumPz_AK42 = 0,sumPz_AK43 = 0,sumPz_AK44 = 0;
   double sumP_AK41 = 0,sumP_AK42 = 0,sumP_AK43 = 0, sumP_AK44 = 0;

   int n_AK41_parts_1 = 0, n_AK41_parts_5 = 0, n_AK41_parts_10 =0; // n_AK41_parts_20=0, n_AK41_parts_40=0, n_AK41_parts_50=0, n_AK41_parts_75=0, n_AK41_parts_100=0;
   int n_AK41_parts_0p1 = 0, n_AK41_parts_0p5 = 0, n_AK41_parts_2 =0, n_AK41_parts_7p5=0, n_AK41_parts_15=0;
   int n_AK42_parts_1 = 0, n_AK42_parts_5 = 0, n_AK42_parts_10 =0; 
   int n_AK42_parts_0p1 = 0, n_AK42_parts_0p5 = 0, n_AK42_parts_2 =0, n_AK42_parts_7p5=0, n_AK42_parts_15=0;
   int n_AK43_parts_1 = 0, n_AK43_parts_5 = 0, n_AK43_parts_10 =0; 
   int n_AK43_parts_0p1 = 0, n_AK43_parts_0p5 = 0, n_AK43_parts_2 =0, n_AK43_parts_7p5=0, n_AK43_parts_15=0;
   int n_AK44_parts_1 = 0, n_AK44_parts_5 = 0, n_AK44_parts_10 =0; 
   int n_AK44_parts_0p1 = 0, n_AK44_parts_0p5 = 0, n_AK44_parts_2 =0, n_AK44_parts_7p5=0, n_AK44_parts_15=0;



   // reclustered AK4 jets where only particles with energy greater than X are considered 
   TLorentzVector AK41_0p1(0,0,0,0); TLorentzVector AK41_0p5(0,0,0,0); TLorentzVector AK41_1(0,0,0,0); 
   TLorentzVector AK41_2(0,0,0,0); TLorentzVector AK41_5(0,0,0,0); TLorentzVector AK41_7p5(0,0,0,0);
   TLorentzVector AK41_10(0,0,0,0); TLorentzVector AK41_15(0,0,0,0);
   TLorentzVector AK41_20(0,0,0,0); //TLorentzVector AK41_40(0,0,0,0);
   //TLorentzVector AK41_50(0,0,0,0); TLorentzVector AK41_75(0,0,0,0); TLorentzVector AK41_100(0,0,0,0);
   TLorentzVector AK42_0p1(0,0,0,0); TLorentzVector AK42_0p5(0,0,0,0); TLorentzVector AK42_1(0,0,0,0); 
   TLorentzVector AK42_2(0,0,0,0); TLorentzVector AK42_5(0,0,0,0); TLorentzVector AK42_7p5(0,0,0,0);
   TLorentzVector AK42_10(0,0,0,0); TLorentzVector AK42_15(0,0,0,0);
   TLorentzVector AK42_20(0,0,0,0); 
   TLorentzVector AK43_0p1(0,0,0,0); TLorentzVector AK43_0p5(0,0,0,0); TLorentzVector AK43_1(0,0,0,0); 
   TLorentzVector AK43_2(0,0,0,0); TLorentzVector AK43_5(0,0,0,0); TLorentzVector AK43_7p5(0,0,0,0);
   TLorentzVector AK43_10(0,0,0,0); TLorentzVector AK43_15(0,0,0,0);
   TLorentzVector AK43_20(0,0,0,0); 
   TLorentzVector AK44_0p1(0,0,0,0); TLorentzVector AK44_0p5(0,0,0,0); TLorentzVector AK44_1(0,0,0,0); 
   TLorentzVector AK44_2(0,0,0,0); TLorentzVector AK44_5(0,0,0,0); TLorentzVector AK44_7p5(0,0,0,0);
   TLorentzVector AK44_10(0,0,0,0); TLorentzVector AK44_15(0,0,0,0);
   TLorentzVector AK44_20(0,0,0,0); 

   for(auto iP = AK41_parts.begin(); iP != AK41_parts.end(); iP++)
   {
      iP->Boost(-AK41_boost.X(),-AK41_boost.Y(), -AK41_boost.Z());
      boostedAK41_Part_TLV.push_back(TLorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E()));
      boostedAK41_Part_LC.push_back(reco::LeafCandidate(+1, reco::Candidate::LorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E())));
      boostedAK41_Part_XYZ.push_back(math::XYZVector( iP->Px(),iP->Py(),iP->Pz() ));
      if(iP->E() > 0.1)
      {
         AK41_0p1+= *iP;
         n_AK41_parts_0p1++;
      }
      if(iP->E() > 0.5)
      {
         AK41_0p5+= *iP;
         n_AK41_parts_0p5++;
      }
      if(iP->E() > 1)
      {
         AK41_1+= *iP;
         n_AK41_parts_1++;
      }
      if(iP->E() > 2)
      {
         AK41_2+= *iP;
         n_AK41_parts_2++;
      }
      if(iP->E() > 5)
      {
         AK41_5+= *iP;
         n_AK41_parts_5++;
      }
      if(iP->E() > 7.5)
      {
         AK41_7p5+= *iP;
         n_AK41_parts_7p5++;
      }
      if(iP->E() > 10)
      {
         AK41_10+= *iP;
         n_AK41_parts_10++;
      }
      if(iP->E() > 15)
      {
         AK41_15+= *iP;
         n_AK41_parts_15++;
      }
      sumPz_AK41+= abs(iP->Pz()); sumP_AK41+= abs(iP->P());
   }
   for(auto iP = AK42_parts.begin(); iP != AK42_parts.end(); iP++)
   {
      iP->Boost(-AK42_boost.X(),-AK42_boost.Y(), -AK42_boost.Z());
      boostedAK42_Part_TLV.push_back(TLorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E()));
      boostedAK42_Part_LC.push_back(reco::LeafCandidate(+1, reco::Candidate::LorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E())));
      boostedAK42_Part_XYZ.push_back(math::XYZVector( iP->Px(),iP->Py(),iP->Pz() ));  
      if(iP->E() > 0.1)
      {
         AK42_0p1+= *iP;
         n_AK42_parts_0p1++;
      }
      if(iP->E() > 0.5)
      {
         AK42_0p5+= *iP;
         n_AK42_parts_0p5++;
      }
      if(iP->E() > 1)
      {
         AK42_1+= *iP;
         n_AK42_parts_1++;
      }
      if(iP->E() > 2)
      {
         AK42_2+= *iP;
         n_AK42_parts_2++;
      }
      if(iP->E() > 5)
      {
         AK42_5+= *iP;
         n_AK42_parts_5++;
      }
      if(iP->E() > 7.5)
      {
         AK42_7p5+= *iP;
         n_AK42_parts_7p5++;
      }
      if(iP->E() > 10)
      {
         AK42_10+= *iP;
         n_AK42_parts_10++;
      }
      if(iP->E() > 15)
      {
         AK42_15+= *iP;
         n_AK42_parts_15++;
      }
      sumPz_AK42+= abs(iP->Pz()); sumP_AK42+= abs(iP->P());
   }
   for(auto iP = AK43_parts.begin(); iP != AK43_parts.end(); iP++)
   {
      iP->Boost(-AK43_boost.X(),-AK43_boost.Y(), -AK43_boost.Z());
      boostedAK43_Part_TLV.push_back(TLorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E()));
      boostedAK43_Part_LC.push_back(reco::LeafCandidate(+1, reco::Candidate::LorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E())));
      boostedAK43_Part_XYZ.push_back(math::XYZVector( iP->Px(),iP->Py(),iP->Pz() )); 
      if(iP->E() > 0.1)
      {
         AK43_0p1+= *iP;
         n_AK43_parts_0p1++;
      }
      if(iP->E() > 0.5)
      {
         AK43_0p5+= *iP;
         n_AK43_parts_0p5++;
      }
      if(iP->E() > 1)
      {
         AK43_1+= *iP;
         n_AK43_parts_1++;
      }
      if(iP->E() > 2)
      {
         AK43_2+= *iP;
         n_AK43_parts_2++;
      }
      if(iP->E() > 5)
      {
         AK43_5+= *iP;
         n_AK43_parts_5++;
      }
      if(iP->E() > 7.5)
      {
         AK43_7p5+= *iP;
         n_AK43_parts_7p5++;
      }
      if(iP->E() > 10)
      {
         AK43_10+= *iP;
         n_AK43_parts_10++;
      }
      if(iP->E() > 15)
      {
         AK43_15+= *iP;
         n_AK43_parts_15++;
      }
      sumPz_AK43 += abs(iP->Pz()); sumP_AK43 += abs(iP->P());   
   }
   for(auto iP = AK44_parts.begin(); iP != AK44_parts.end(); iP++)
   {
      iP->Boost(-AK44_boost.X(),-AK44_boost.Y(), -AK44_boost.Z());
      boostedAK44_Part_TLV.push_back(TLorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E()));
      boostedAK44_Part_LC.push_back(reco::LeafCandidate(+1, reco::Candidate::LorentzVector(iP->Px(),iP->Py(),iP->Pz(),iP->E())));
      boostedAK44_Part_XYZ.push_back(math::XYZVector( iP->Px(),iP->Py(),iP->Pz() )); 
      if(iP->E() > 0.1)
      {
         AK44_0p1+= *iP;
         n_AK44_parts_0p1++;
      }
      if(iP->E() > 0.5)
      {
         AK44_0p5+= *iP;
         n_AK44_parts_0p5++;
      }
      if(iP->E() > 1)
      {
         AK44_1+= *iP;
         n_AK44_parts_1++;
      }
      if(iP->E() > 2)
      {
         AK44_2+= *iP;
         n_AK44_parts_2++;
      }
      if(iP->E() > 5)
      {
         AK44_5+= *iP;
         n_AK44_parts_5++;
      }
      if(iP->E() > 7.5)
      {
         AK44_7p5+= *iP;
         n_AK44_parts_7p5++;
      }
      if(iP->E() > 10)
      {
         AK44_10+= *iP;
         n_AK44_parts_10++;
      }
      if(iP->E() > 15)
      {
         AK44_15+= *iP;
         n_AK44_parts_15++;
      }
      sumPz_AK44 += abs(iP->Pz()); sumP_AK44 += abs(iP->P());   
   }

   ////vectors to get angles
        
   TVector3 AK4_jet1(jetsFJ_jet[0].px(),jetsFJ_jet[0].py(),jetsFJ_jet[0].pz());
   TVector3 AK4_jet2(jetsFJ_jet[1].px(),jetsFJ_jet[1].py(),jetsFJ_jet[1].pz());
   TVector3 AK4_jet3(jetsFJ_jet[2].px(),jetsFJ_jet[2].py(),jetsFJ_jet[2].pz());
   TVector3 AK4_jet4(jetsFJ_jet[3].px(),jetsFJ_jet[3].py(),jetsFJ_jet[3].pz());

   //fill superjet variables here ...
   //SJ mass variables

   treeVars["SJ_mass"]    = sqrt(pow(superJetE,2)-pow(superJetpx,2)-pow(superJetpy,2)-pow(superJetpz,2)); 
   treeVars["SJ_mass_25"] = sqrt(pow(SJ_25_E,2)-pow(SJ_25_px,2)-pow(SJ_25_py,2)-pow(SJ_25_pz,2)); 
   treeVars["SJ_mass_50"] = sqrt(pow(SJ_50_E,2)-pow(SJ_50_px,2)-pow(SJ_50_py,2)-pow(SJ_50_pz,2)); 
   treeVars["SJ_mass_100"] = sqrt(pow(SJ_100_E,2)-pow(SJ_100_px,2)-pow(SJ_100_py,2)-pow(SJ_100_pz,2)); 
   treeVars["SJ_mass_150"] = sqrt(pow(SJ_150_E,2)-pow(SJ_150_px,2)-pow(SJ_150_py,2)-pow(SJ_150_pz,2)); 
   treeVars["SJ_mass_200"] = sqrt(pow(SJ_200_E,2)-pow(SJ_200_px,2)-pow(SJ_200_py,2)-pow(SJ_200_pz,2)); 
   treeVars["SJ_mass_300"] = sqrt(pow(SJ_300_E,2)-pow(SJ_300_px,2)-pow(SJ_300_py,2)-pow(SJ_300_pz,2)); 

   treeVars["SJ_mass_400"] = sqrt(pow(SJ_400_E,2)-pow(SJ_400_px,2)-pow(SJ_400_py,2)-pow(SJ_400_pz,2)); 
   treeVars["SJ_mass_500"] = sqrt(pow(SJ_500_E,2)-pow(SJ_500_px,2)-pow(SJ_500_py,2)-pow(SJ_500_pz,2)); 
   treeVars["SJ_mass_800"] = sqrt(pow(SJ_800_E,2)-pow(SJ_800_px,2)-pow(SJ_800_py,2)-pow(SJ_800_pz,2)); 
   treeVars["SJ_mass_1000"] = sqrt(pow(SJ_1000_E,2)-pow(SJ_1000_px,2)-pow(SJ_1000_py,2)-pow(SJ_1000_pz,2)); 

   double offsetInts = 0.5;

   //SJ nAK4 variables
   treeVars["SJ_nAK4_25"] = SJ_nAK4_25_ + offsetInts; 
   treeVars["SJ_nAK4_50"] = SJ_nAK4_50_ + offsetInts; 
   treeVars["SJ_nAK4_100"] = SJ_nAK4_100_ + offsetInts; 
   treeVars["SJ_nAK4_150"] = SJ_nAK4_150_ + offsetInts; 
   treeVars["SJ_nAK4_200"] = SJ_nAK4_200_ + offsetInts; 
   treeVars["SJ_nAK4_300"] = SJ_nAK4_300_ + offsetInts; 
   treeVars["SJ_nAK4_400"] = SJ_nAK4_400_ + offsetInts; 
   treeVars["SJ_nAK4_500"] = SJ_nAK4_500_ + offsetInts; 
   treeVars["SJ_nAK4_800"] = SJ_nAK4_800_ + offsetInts; 
   treeVars["SJ_nAK4_1000"] = SJ_nAK4_1000_ + offsetInts; 


   treeVars["AK41_nDaughters"] = jetsFJ_jet[0].constituents().size() + offsetInts; 
   treeVars["AK42_nDaughters"] = jetsFJ_jet[1].constituents().size() + offsetInts; 
   treeVars["AK43_nDaughters"] = jetsFJ_jet[2].constituents().size() + offsetInts; 
   treeVars["AK44_nDaughters"] = jetsFJ_jet[3].constituents().size() + offsetInts; 


   //AK4 jet mass combinations
   AK4_m1[nSuperJets] = jetsFJ_jet[0].m();   
   AK4_m2[nSuperJets] = jetsFJ_jet[1].m(); 
   AK4_m3[nSuperJets] = jetsFJ_jet[2].m(); 
   AK4_m4[nSuperJets] = jetsFJ_jet[3].m(); 

   AK41_E_tree[nSuperJets] = jetsFJ_jet[0].E(); 
   AK42_E_tree[nSuperJets] = jetsFJ_jet[1].E(); 
   AK43_E_tree[nSuperJets] = jetsFJ_jet[2].E(); 
   AK44_E_tree[nSuperJets] = jetsFJ_jet[3].E(); 


   AK4_m12[nSuperJets] = sqrt( pow(jetsFJ_jet[0].E() + jetsFJ_jet[1].E() ,2) - pow(jetsFJ_jet[0].px() + jetsFJ_jet[1].px(),2) - pow(jetsFJ_jet[0].py() + jetsFJ_jet[1].py(),2)- pow(jetsFJ_jet[0].pz() + jetsFJ_jet[1].pz(),2));   
   AK4_m13[nSuperJets] = sqrt( pow(jetsFJ_jet[0].E() + jetsFJ_jet[2].E() ,2) - pow(jetsFJ_jet[0].px() + jetsFJ_jet[2].px(),2) - pow(jetsFJ_jet[0].py() + jetsFJ_jet[2].py(),2)- pow(jetsFJ_jet[0].pz() + jetsFJ_jet[2].pz(),2));   
   AK4_m14[nSuperJets] = sqrt( pow(jetsFJ_jet[0].E() + jetsFJ_jet[3].E() ,2) - pow(jetsFJ_jet[0].px() + jetsFJ_jet[3].px(),2) - pow(jetsFJ_jet[0].py() + jetsFJ_jet[3].py(),2)- pow(jetsFJ_jet[0].pz() + jetsFJ_jet[3].pz(),2));   
   AK4_m23[nSuperJets] = sqrt( pow(jetsFJ_jet[2].E() + jetsFJ_jet[1].E() ,2) - pow(jetsFJ_jet[2].px() + jetsFJ_jet[1].px(),2) - pow(jetsFJ_jet[2].py() + jetsFJ_jet[1].py(),2)- pow(jetsFJ_jet[2].pz() + jetsFJ_jet[1].pz(),2));   
   AK4_m24[nSuperJets] = sqrt( pow(jetsFJ_jet[3].E() + jetsFJ_jet[1].E() ,2) - pow(jetsFJ_jet[3].px() + jetsFJ_jet[1].px(),2) - pow(jetsFJ_jet[3].py() + jetsFJ_jet[1].py(),2)- pow(jetsFJ_jet[3].pz() + jetsFJ_jet[1].pz(),2));   
   AK4_m34[nSuperJets] = sqrt( pow(jetsFJ_jet[2].E() + jetsFJ_jet[3].E() ,2) - pow(jetsFJ_jet[2].px() + jetsFJ_jet[3].px(),2) - pow(jetsFJ_jet[2].py() + jetsFJ_jet[3].py(),2)- pow(jetsFJ_jet[2].pz() + jetsFJ_jet[3].pz(),2));   

   AK4_m123[nSuperJets] =sqrt( pow(jetsFJ_jet[0].E() + jetsFJ_jet[1].E() + jetsFJ_jet[2].E() ,2) - pow(jetsFJ_jet[0].px() + jetsFJ_jet[1].px() + jetsFJ_jet[2].px(),2) - pow(jetsFJ_jet[0].py() + jetsFJ_jet[1].py() + jetsFJ_jet[2].py(),2)- pow(jetsFJ_jet[0].pz() + jetsFJ_jet[1].pz() + jetsFJ_jet[2].pz(),2));   
   AK4_m124[nSuperJets] =sqrt( pow(jetsFJ_jet[0].E() + jetsFJ_jet[1].E() + jetsFJ_jet[3].E() ,2) - pow(jetsFJ_jet[0].px() + jetsFJ_jet[1].px() + jetsFJ_jet[3].px(),2) - pow(jetsFJ_jet[0].py() + jetsFJ_jet[1].py() + jetsFJ_jet[3].py(),2)- pow(jetsFJ_jet[0].pz() + jetsFJ_jet[1].pz() + jetsFJ_jet[3].pz(),2));   
   AK4_m134[nSuperJets] =sqrt( pow(jetsFJ_jet[0].E() + jetsFJ_jet[2].E() + jetsFJ_jet[3].E() ,2) - pow(jetsFJ_jet[0].px() + jetsFJ_jet[2].px() + jetsFJ_jet[3].px(),2) - pow(jetsFJ_jet[0].py() + jetsFJ_jet[2].py() + jetsFJ_jet[3].py(),2)- pow(jetsFJ_jet[0].pz() + jetsFJ_jet[2].pz() + jetsFJ_jet[3].pz(),2));   
   AK4_m234[nSuperJets] =sqrt( pow(jetsFJ_jet[1].E() + jetsFJ_jet[2].E() + jetsFJ_jet[3].E() ,2) - pow(jetsFJ_jet[1].px() + jetsFJ_jet[2].px() + jetsFJ_jet[3].px(),2) - pow(jetsFJ_jet[1].py() + jetsFJ_jet[2].py() + jetsFJ_jet[3].py(),2)- pow(jetsFJ_jet[1].pz() + jetsFJ_jet[2].pz() + jetsFJ_jet[3].pz(),2));   

   AK4_m1234[nSuperJets] =sqrt( pow(jetsFJ_jet[0].E() + jetsFJ_jet[1].E() + jetsFJ_jet[2].E()+jetsFJ_jet[3].E() ,2) - pow(jetsFJ_jet[0].px() + jetsFJ_jet[1].px() + jetsFJ_jet[2].px()+jetsFJ_jet[3].px(),2) - pow(jetsFJ_jet[0].py() + jetsFJ_jet[1].py() + jetsFJ_jet[2].py()+jetsFJ_jet[3].py(),2)- pow(jetsFJ_jet[0].pz() + jetsFJ_jet[1].pz() + jetsFJ_jet[2].pz()+jetsFJ_jet[3].pz(),2));   

   AK4_theta12[nSuperJets] = cos(abs(AK4_jet1.Angle(AK4_jet2)));   
   AK4_theta13[nSuperJets] = cos(abs(AK4_jet1.Angle(AK4_jet3)));
   AK4_theta14[nSuperJets] = cos(abs(AK4_jet1.Angle(AK4_jet4)));
   AK4_theta23[nSuperJets] = cos(abs(AK4_jet2.Angle(AK4_jet3)));
   AK4_theta24[nSuperJets] = cos(abs(AK4_jet2.Angle(AK4_jet4)));
   AK4_theta34[nSuperJets] = cos(abs(AK4_jet3.Angle(AK4_jet4)));

   EventShapeVariables eventShapesAK41( boostedAK41_Part_XYZ );
   Thrust thrustCalculatorAK41( boostedAK41_Part_LC.begin(), boostedAK41_Part_LC.end() );
   double fwmAK41[5] = { 0.0, 0.0 ,0.0 ,0.0,0.0};
   FWMoments( boostedAK41_Part_TLV, fwmAK41); 

   EventShapeVariables eventShapesAK42( boostedAK42_Part_XYZ );
   Thrust thrustCalculatorAK42( boostedAK42_Part_LC.begin(), boostedAK42_Part_LC.end() );
   double fwmAK42[5] = { 0.0, 0.0 ,0.0 ,0.0,0.0};
   FWMoments( boostedAK42_Part_TLV, fwmAK42); 

   EventShapeVariables eventShapesAK43( boostedAK43_Part_XYZ );
   Thrust thrustCalculatorAK43( boostedAK43_Part_LC.begin(), boostedAK43_Part_LC.end() );
   double fwmAK43[5] = { 0.0, 0.0 ,0.0 ,0.0,0.0};
   FWMoments( boostedAK43_Part_TLV, fwmAK43); 

   EventShapeVariables eventShapesAK44( boostedAK44_Part_XYZ );
   Thrust thrustCalculatorAK44( boostedAK44_Part_LC.begin(), boostedAK44_Part_LC.end() );
   double fwmAK44[5] = { 0.0, 0.0 ,0.0 ,0.0,0.0};
   FWMoments( boostedAK44_Part_TLV, fwmAK44); 



   //AK4 jet boosted information - boost reclustered AK4 jets into their COM and look at BES variables, ndaughters, nsubjettiness
   AK41_ndaughters[nSuperJets] = jetsFJ_jet[0].constituents().size();  
   AK41_nsubjets[nSuperJets] = jetsFJ_jet[0].n_exclusive_subjets(0.2); 
   AK41_thrust[nSuperJets] = thrustCalculatorAK41.thrust();
   AK41_sphericity[nSuperJets] = eventShapesAK41.sphericity();
   AK41_asymmetry[nSuperJets] = sumPz_AK41/sumP_AK41; 
   AK41_isotropy[nSuperJets] = eventShapesAK41.isotropy();
   AK41_aplanarity[nSuperJets] = eventShapesAK41.aplanarity();
   AK41_FW1[nSuperJets] = fwmAK41[1]; 
   AK41_FW2[nSuperJets] = fwmAK41[2]; 
   AK41_FW3[nSuperJets] = fwmAK41[3]; 
   AK41_FW4[nSuperJets] = fwmAK41[4]; 

   AK42_ndaughters[nSuperJets] = jetsFJ_jet[1].constituents().size(); 
   AK42_nsubjets[nSuperJets] = jetsFJ_jet[1].exclusive_subjets(0.2).size();
   AK42_thrust[nSuperJets] = thrustCalculatorAK42.thrust(); 
   AK42_sphericity[nSuperJets] = eventShapesAK42.sphericity();
   AK42_asymmetry[nSuperJets] = sumPz_AK42/sumP_AK42; 
   AK42_isotropy[nSuperJets] = eventShapesAK42.isotropy();
   AK42_aplanarity[nSuperJets] = eventShapesAK42.aplanarity();
   AK42_FW1[nSuperJets] = fwmAK42[1]; 
   AK42_FW2[nSuperJets] = fwmAK42[2]; 
   AK42_FW3[nSuperJets] = fwmAK42[3]; 
   AK42_FW4[nSuperJets] = fwmAK42[4]; 

   AK43_ndaughters[nSuperJets] = jetsFJ_jet[2].constituents().size(); 
   AK43_nsubjets[nSuperJets] = jetsFJ_jet[2].exclusive_subjets(0.2).size();
   AK43_thrust[nSuperJets] = thrustCalculatorAK43.thrust();
   AK43_sphericity[nSuperJets] = eventShapesAK43.sphericity();
   AK43_asymmetry[nSuperJets] = sumPz_AK43/sumP_AK43; 
   AK43_isotropy[nSuperJets] = eventShapesAK43.isotropy();
   AK43_aplanarity[nSuperJets] = eventShapesAK43.aplanarity();
   AK43_FW1[nSuperJets] = fwmAK43[1]; 
   AK43_FW2[nSuperJets] = fwmAK43[2]; 
   AK43_FW3[nSuperJets] = fwmAK43[3]; 
   AK43_FW4[nSuperJets] = fwmAK43[4]; 

   EventShapeVariables eventShapes( boostedSuperJetPart_XYZ );
   Thrust thrustCalculator( boostedSuperJetPart_LC.begin(), boostedSuperJetPart_LC.end() );
   double fwm[5] = { 0.0, 0.0 ,0.0 ,0.0,0.0};
   FWMoments( boostedSuperJetPart_TLV, fwm); 


   SJ_thrust[nSuperJets] = thrustCalculator.thrust();
   SJ_sphericity[nSuperJets] = eventShapes.sphericity();
   SJ_asymmetry[nSuperJets] = sumPz/sumP; 
   SJ_isotropy[nSuperJets] = eventShapes.isotropy();
   SJ_aplanarity[nSuperJets] = eventShapes.aplanarity();
   SJ_FW1[nSuperJets] = fwm[1]; 
   SJ_FW2[nSuperJets] = fwm[2]; 
   SJ_FW3[nSuperJets] = fwm[3]; 
   SJ_FW4[nSuperJets] = fwm[4];


   //AK4 jet mass combinations
   treeVars["AK4_m1"] = AK4_m1[nSuperJets]; 
   treeVars["AK4_m2"] = AK4_m2[nSuperJets]; 
   treeVars["AK4_m3"] = AK4_m3[nSuperJets]; 
   treeVars["AK4_m4"] = AK4_m4[nSuperJets]; 

   treeVars["AK41_E"] = AK41_E_tree[nSuperJets]; 
   treeVars["AK42_E"] = AK42_E_tree[nSuperJets]; 
   treeVars["AK43_E"] = AK43_E_tree[nSuperJets]; 
   treeVars["AK44_E"] = AK44_E_tree[nSuperJets]; 


   treeVars["AK4_m12"] = AK4_m12[nSuperJets];   
   treeVars["AK4_m13"] = AK4_m13[nSuperJets];   
   treeVars["AK4_m14"] = AK4_m14[nSuperJets];   
   treeVars["AK4_m23"] = AK4_m23[nSuperJets];   
   treeVars["AK4_m24"] = AK4_m24[nSuperJets];   
   treeVars["AK4_m34"] = AK4_m34[nSuperJets];   

   treeVars["AK4_m123"] = AK4_m123[nSuperJets];   
   treeVars["AK4_m124"] =AK4_m124[nSuperJets];   
   treeVars["AK4_m134"] =AK4_m134[nSuperJets];   
   treeVars["AK4_m234"] = AK4_m234[nSuperJets];   

   treeVars["AK4_m1234"] = AK4_m1234[nSuperJets];   
   

   //AK4 jet angles 
   treeVars["AK4_theta12"] = AK4_theta12[nSuperJets];   
   treeVars["AK4_theta13"] = AK4_theta13[nSuperJets];
   treeVars["AK4_theta14"] = AK4_theta14[nSuperJets];
   treeVars["AK4_theta23"] = AK4_theta23[nSuperJets];
   treeVars["AK4_theta24"] = AK4_theta24[nSuperJets];
   treeVars["AK4_theta34"] = AK4_theta34[nSuperJets];

   //AK4 jet boosted information - boost reclustered AK4 jets into their COM and look at BES variables, ndaughters, nsubjettiness
   treeVars["AK41_nsubjets"] = jetsFJ_jet[0].n_exclusive_subjets(0.2) + offsetInts; 
   treeVars["AK41_thrust"] = thrustCalculatorAK41.thrust();
   treeVars["AK41_sphericity"] = eventShapesAK41.sphericity();
   treeVars["AK41_asymmetry"] = sumPz_AK41/sumP_AK41; 
   treeVars["AK41_isotropy"] = eventShapesAK41.isotropy();
   treeVars["AK41_aplanarity"] = eventShapesAK41.aplanarity();
   treeVars["AK41_FW1"] = fwmAK41[1]; 
   treeVars["AK41_FW2"] = fwmAK41[2]; 
   treeVars["AK41_FW3"] = fwmAK41[3]; 
   treeVars["AK41_FW4"] = fwmAK41[4]; 

   treeVars["AK42_nsubjets"] = jetsFJ_jet[1].exclusive_subjets(0.2).size() + offsetInts;
   treeVars["AK42_thrust"] = thrustCalculatorAK42.thrust(); 
   treeVars["AK42_sphericity"] = eventShapesAK42.sphericity();
   treeVars["AK42_asymmetry"] = sumPz_AK42/sumP_AK42; 
   treeVars["AK42_isotropy"] = eventShapesAK42.isotropy();
   treeVars["AK42_aplanarity"] = eventShapesAK42.aplanarity();
   treeVars["AK42_FW1"] = fwmAK42[1]; 
   treeVars["AK42_FW2"] = fwmAK42[2]; 
   treeVars["AK42_FW3"] = fwmAK42[3]; 
   treeVars["AK42_FW4"] = fwmAK42[4]; 

   treeVars["AK43_nsubjets"] = jetsFJ_jet[2].exclusive_subjets(0.2).size() + offsetInts;
   treeVars["AK43_thrust"] = thrustCalculatorAK43.thrust();
   treeVars["AK43_sphericity"] = eventShapesAK43.sphericity();
   treeVars["AK43_asymmetry"] = sumPz_AK43/sumP_AK43; 
   treeVars["AK43_isotropy"] = eventShapesAK43.isotropy();
   treeVars["AK43_aplanarity"] = eventShapesAK43.aplanarity();
   treeVars["AK43_FW1"] = fwmAK43[1]; 
   treeVars["AK43_FW2"] = fwmAK43[2]; 
   treeVars["AK43_FW3"] = fwmAK43[3]; 
   treeVars["AK43_FW4"] = fwmAK43[4]; 

   treeVars["AK44_nsubjets"] = jetsFJ_jet[3].exclusive_subjets(0.2).size() + offsetInts;
   treeVars["AK44_thrust"] = thrustCalculatorAK44.thrust();
   treeVars["AK44_sphericity"] = eventShapesAK44.sphericity();
   treeVars["AK44_asymmetry"] = sumPz_AK44/sumP_AK44; 
   treeVars["AK44_isotropy"] = eventShapesAK44.isotropy();
   treeVars["AK44_aplanarity"] = eventShapesAK44.aplanarity();
   treeVars["AK44_FW1"] = fwmAK44[1]; 
   treeVars["AK44_FW2"] = fwmAK44[2]; 
   treeVars["AK44_FW3"] = fwmAK44[3]; 
   treeVars["AK44_FW4"] = fwmAK44[4]; 

   //Full SJ BES variablesf

   treeVars["SJ_thrust"] = thrustCalculator.thrust();
   treeVars["SJ_sphericity"] = eventShapes.sphericity();
   treeVars["SJ_asymmetry"] = sumPz/sumP; 
   treeVars["SJ_isotropy"] = eventShapes.isotropy();
   treeVars["SJ_aplanarity"] = eventShapes.aplanarity();
   treeVars["SJ_FW1"] = fwm[1]; 
   treeVars["SJ_FW2"] = fwm[2]; 
   treeVars["SJ_FW3"] = fwm[3]; 
   treeVars["SJ_FW4"] = fwm[4]; 

  if(testNewVars)
  {

     treeVars["SJ_AK4_frac_10"] = 1.0*SJ_nAK4_10_ / SJ_nAK4_1;
     treeVars["SJ_AK4_frac_25"] = 1.0*SJ_nAK4_25_ / SJ_nAK4_1;
     treeVars["SJ_AK4_frac_50"] = 1.0*SJ_nAK4_50_ / SJ_nAK4_1;
     treeVars["SJ_AK4_frac_75"] = 1.0*SJ_nAK4_75_ / SJ_nAK4_1;
     treeVars["SJ_AK4_frac_100"] = 1.0*SJ_nAK4_100_/ SJ_nAK4_1;
     treeVars["SJ_AK4_frac_200"] = 1.0*SJ_nAK4_200_/ SJ_nAK4_1;
     treeVars["SJ_AK4_frac_300"] = 1.0*SJ_nAK4_300_/ SJ_nAK4_1;
     treeVars["SJ_AK4_frac_500"] = 1.0*SJ_nAK4_500_ / SJ_nAK4_1;
     treeVars["SJ_AK4_frac_800"] = 1.0*SJ_nAK4_800_/ SJ_nAK4_1;

     treeVars["AK41_daughters_frac_0p1"] = 1.0*n_AK41_parts_0p1  /AK41_parts.size();   
     treeVars["AK41_daughters_frac_0p5"] = 1.0*n_AK41_parts_0p5  /AK41_parts.size();   
     treeVars["AK41_daughters_frac_1"] = 1.0*n_AK41_parts_1  /AK41_parts.size();   
     treeVars["AK41_daughters_frac_2"] = 1.0*n_AK41_parts_2  /AK41_parts.size();   
     treeVars["AK41_daughters_frac_5"] = 1.0*n_AK41_parts_5  /AK41_parts.size(); 
     treeVars["AK41_daughters_frac_7p5"] = 1.0*n_AK41_parts_7p5 /AK41_parts.size();     
     treeVars["AK41_daughters_frac_10"] = 1.0*n_AK41_parts_10  /AK41_parts.size();   
     treeVars["AK41_daughters_frac_15"] = 1.0*n_AK41_parts_15 /AK41_parts.size();   

     treeVars["AK41_mass_0p1"] = AK41_0p1.M();
     treeVars["AK41_mass_0p5"] = AK41_0p5.M();
     treeVars["AK41_mass_1"] = AK41_1.M();
     treeVars["AK41_mass_2"] = AK41_2.M();
     //treeVars["AK41_mass_5"] = AK41_5.M();
     treeVars["AK41_mass_7p5"] = AK41_7p5.M();
     treeVars["AK41_mass_10"] = AK41_10.M();
     treeVars["AK41_mass_15"] = AK41_15.M();

     treeVars["AK42_daughters_frac_0p1"] = 1.0*n_AK42_parts_0p1  /AK42_parts.size();   
     treeVars["AK42_daughters_frac_0p5"] = 1.0*n_AK42_parts_0p5  /AK42_parts.size();   
     treeVars["AK42_daughters_frac_1"] = 1.0*n_AK42_parts_1  /AK42_parts.size();   
     treeVars["AK42_daughters_frac_2"] = 1.0*n_AK42_parts_2  /AK42_parts.size();   
     treeVars["AK42_daughters_frac_5"] = 1.0*n_AK42_parts_5  /AK42_parts.size(); 
     treeVars["AK42_daughters_frac_7p5"] = 1.0*n_AK42_parts_7p5 /AK42_parts.size();     
     treeVars["AK42_daughters_frac_10"] = 1.0*n_AK42_parts_10  /AK42_parts.size();   
     treeVars["AK42_daughters_frac_15"] = 1.0*n_AK42_parts_15 /AK42_parts.size();   

     treeVars["AK42_mass_0p1"] = AK42_0p1.M();
     treeVars["AK42_mass_0p5"] = AK42_0p5.M();
     treeVars["AK42_mass_1"] = AK42_1.M();
     treeVars["AK42_mass_2"] = AK42_2.M();
     //treeVars["AK42_mass_5"] = AK42_5.M();
     treeVars["AK42_mass_7p5"] = AK42_7p5.M();
     treeVars["AK42_mass_10"] = AK42_10.M();
     treeVars["AK42_mass_15"] = AK42_15.M();

     treeVars["AK43_daughters_frac_0p1"] = 1.0*n_AK43_parts_0p1  /AK43_parts.size();   
     treeVars["AK43_daughters_frac_0p5"] = 1.0*n_AK43_parts_0p5  /AK43_parts.size();   
     treeVars["AK43_daughters_frac_1"] = 1.0*n_AK43_parts_1  /AK43_parts.size();   
     treeVars["AK43_daughters_frac_2"] = 1.0*n_AK43_parts_2  /AK43_parts.size();   
     treeVars["AK43_daughters_frac_5"] = 1.0*n_AK43_parts_5  /AK43_parts.size(); 
     treeVars["AK43_daughters_frac_7p5"] = 1.0*n_AK43_parts_7p5 /AK43_parts.size();     
     treeVars["AK43_daughters_frac_10"] = 1.0*n_AK43_parts_10  /AK43_parts.size();   
     treeVars["AK43_daughters_frac_15"] = 1.0*n_AK43_parts_15 /AK43_parts.size();   

     treeVars["AK43_mass_0p1"] = AK43_0p1.M();
     treeVars["AK43_mass_0p5"] = AK43_0p5.M();
     treeVars["AK43_mass_1"] = AK43_1.M();
     treeVars["AK43_mass_2"] = AK43_2.M();
     //treeVars["AK43_mass_5"] = AK43_5.M();
     treeVars["AK43_mass_7p5"] = AK43_7p5.M();
     treeVars["AK43_mass_10"] = AK43_10.M();
     treeVars["AK43_mass_15"] = AK43_15.M();

     treeVars["AK44_daughters_frac_0p1"] = 1.0*n_AK44_parts_0p1  /AK44_parts.size();   
     treeVars["AK44_daughters_frac_0p5"] = 1.0*n_AK44_parts_0p5  /AK44_parts.size();   
     treeVars["AK44_daughters_frac_1"] = 1.0*n_AK44_parts_1  /AK44_parts.size();   
     treeVars["AK44_daughters_frac_2"] = 1.0*n_AK44_parts_2  /AK44_parts.size();   
     treeVars["AK44_daughters_frac_5"] = 1.0*n_AK44_parts_5  /AK44_parts.size(); 
     treeVars["AK44_daughters_frac_7p5"] = 1.0*n_AK44_parts_7p5 /AK44_parts.size();     
     treeVars["AK44_daughters_frac_10"] = 1.0*n_AK44_parts_10  /AK44_parts.size();   
     treeVars["AK44_daughters_frac_15"] = 1.0*n_AK44_parts_15 /AK44_parts.size();   

     treeVars["AK44_mass_0p1"] = AK44_0p1.M();
     treeVars["AK44_mass_0p5"] = AK44_0p5.M();
     treeVars["AK44_mass_1"] = AK44_1.M();
     treeVars["AK44_mass_2"] = AK44_2.M();
     //treeVars["AK44_mass_5"] = AK44_5.M();
     treeVars["AK44_mass_7p5"] = AK44_7p5.M();
     treeVars["AK44_mass_10"] = AK44_10.M();
     treeVars["AK44_mass_15"] = AK44_15.M();

  }
/*
 std::cout << "treeVars: " << std::endl;
   for (const auto& pair : treeVars) {
    std::cout << pair.first << ": " << pair.second << std::endl;
   }
*/
  

   ///////////////////// BESvars for tree /////////////
   /////////// fill input variables to NN /////////////
   for (auto i = treeVars.begin(); i!=treeVars.end(); i++)
   {
      //std::cout<< i->first << " "  << i->second << std::endl;
      if (  (i->second  != i->second ) || ( isinf(i->second   )  ) )
      {
           i->second = 0;      //return false; // RETURN cut - event variables are NaN or inf
      }
      else if ( abs( i->second+999.99 ) < 1.0e-10 ) return false;
   }
   return true;

}



// main analyzer function: pre-select events, recluster superjets, calculate all variables from superjets
void selectionStudier_200::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   const edm::EventAuxiliary& aux = iEvent.eventAuxiliary();
   //runNum       = aux.run();
   //lumiBlockNum = aux.luminosityBlock();
   eventNumber       = aux.event();

   if(_verbose)std::cout << " -----------------Starting Event ----------------- " << std::endl;

   edm::Handle< std::vector<reco::GenParticle> > genPartCollection; // this will be used several times throughout the code

   ////////////////////////////////// 
   ///////// Apply Triggers /////////
   //////////////////////////////////


   edm::Handle<edm::TriggerResults> triggerBits;
   iEvent.getByToken(triggerBits_, triggerBits);

   const edm::TriggerNames &names = iEvent.triggerNames(*triggerBits);
   
   passesPFJet = false;
   passesPFHT  = false;

   for(auto iT = triggers.begin(); iT != triggers.end(); iT++)
   {
      std::string trigname = *iT;
      if(debug)std::cout << "Looking for the " << trigname << " trigger." << std::endl; 

      //bool pass = false;
      for (unsigned int i = 0; i < triggerBits->size(); ++i) 
      {
         const std::string name = names.triggerName(i);
         const bool accept = triggerBits->accept(i);
         if ((name.find(trigname) != std::string::npos) &&(accept))
         {

            //std::cout << "trigname is " << trigname << std::endl;
            if( ( trigname == "HLT_PFJet500_v") || (trigname == "HLT_PFJet450_v") ) passesPFJet = true;
            else if( ( trigname == "HLT_PFHT900_v") || (trigname == "HLT_PFHT1050_v") ) passesPFHT = true;

            if(debug)std::cout << "Found the " << *iT << " trigger." << std::endl;
            //pass =true;
         }
      } 
      //if(!pass)
      //{  
      //    return; // if any of the triggers aren't found, skip event   return CUT
      //}
   }
   

   if(_verbose)std::cout << "In analyze" << std::endl;


   /////////////////////////////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////////
   ///////////////////////////////////   _Background MC area_ //////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////////


   if ((runType.find("MC") != std::string::npos) || (runType.find("Suu")!=std::string::npos ) )
   {

      if(_verbose)std::cout << "before pileup" << std::endl;


      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /////////////////////////////////////////////////////////////////////_pileup_//////////////////////////////////////////////////////////////
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //                                              get the pileup weight for this event
      if(doPUSF)
      {
         edm::Handle<std::vector<PileupSummaryInfo> > PupInfo;
         iEvent.getByToken(puSummaryToken_, PupInfo);

         for (auto const& v : *PupInfo)
         {
            ntrueInt = v.getTrueNumInteractions();
            PU_eventWeight_nom  = PUjson_year->evaluate( {std::real(ntrueInt),"nominal"});

            if ((PU_eventWeight_nom != PU_eventWeight_nom) || (std::isinf(PU_eventWeight_nom)))
            {
               std::cout << "Found bad PUSF: val/ntrueInt = " << PU_eventWeight_nom << "/" << ntrueInt <<std::endl;
            }
         }  
      }

   }  

   edm::Handle<double> rho;
   iEvent.getByToken(m_rho_token, rho);




   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   ///////////////////////////////////////////////////////////////////_prefiring_/////////////////////////////////////////////////////////////
   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   edm::Handle< double > theprefweight;
   iEvent.getByToken(prefweight_token, theprefweight ) ;
   prefiringWeight_nom =(*theprefweight);




   if(_verbose)std::cout << "before AK4 jets" << std::endl;
   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////////////_AK4 jets_/////////////////////////////////////////////////////////////
   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   totHT = 0;
   lab_nAK4 = 0;
   nAK4 = 0;
   std::vector<TLorentzVector> leadAK4Jets;

   JME::JetResolution resolution_AK4               = JME::JetResolution::get(iSetup, "AK4PFchs_pt");                          //load JER stuff from global tag
   JME::JetResolutionScaleFactor resolution_sf_AK4 = JME::JetResolutionScaleFactor::get(iSetup, "AK4PFchs");

   edm::Handle<std::vector<pat::Jet>> smallJets;
   iEvent.getByToken(jetToken_, smallJets);


   //// tight b-tag WP variables
   double MC_tagged = 1.0,  MC_notTagged = 1.0, data_tagged = 1.0, data_notTagged = 1.0; //these are for the bTagging SFs

   //// med b-tag WP variables
   double MC_tagged_med = 1.0,  MC_notTagged_med = 1.0, data_tagged_med = 1.0, data_notTagged_med = 1.0; //these are for the bTagging SFs

   // loop over AK4 jets: correct with JECs and JERs, get information for selection, save kinematic information, and calculate b-tagging event weights
   for(auto iJet = smallJets->begin(); iJet != smallJets->end(); iJet++) 
   {

      double AK4_sf_total = 1.0;

      if( (iJet->pt() < 10. ) || (!(iJet->isPFJet())) || ( abs(iJet->eta()) > 2.5 )) continue;   //don't even bother with these jets, lost causes

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //////////   _JET ENERGY RESOLUTION STUFF //////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////

      if(doJER)
      {
         if ((runType.find("MC") != std::string::npos) || (runType.find("Suu") != std::string::npos) )
         {
            double AK4_JER_corr_factor = 1.0; // this won't be touched for data

            if(_verbose)std::cout << "doing JER" << std::endl;

            double sJER     = -9999.;    //JER scale factor
            double sigmaJER = -9999.;    //this is the "resolution" you get from the scale factors 
            
            JME::JetParameters parameters_1;
            parameters_1.setJetPt(iJet->pt());
            parameters_1.setJetEta(iJet->eta());
            parameters_1.setRho(*rho);
            sigmaJER = resolution_AK4.getResolution(parameters_1);   //pT resolution

            //JME::JetParameters
            JME::JetParameters parameters;
            parameters.setJetPt(iJet->pt());
            parameters.setJetEta(iJet->eta());
            sJER =  resolution_sf_AK4.getScaleFactor(parameters  );  //{{JME::Binning::JetEta, iJet->eta()}});
            

            const reco::GenJet *genJet = iJet->genJet();
            if( genJet)   // try the first technique
            {
               AK4_JER_corr_factor = 1 + (sJER - 1)*(iJet->pt()-genJet->pt())/iJet->pt();
            }
            else   // if no gen jet is matched, try the second technique
            {
               randomNum->SetSeed( abs(static_cast<int>(iJet->phi()*1e4)) );
               double JERrand = randomNum->Gaus(0.0, sigmaJER);
               //double JERrand = 1.0 + sigmaJER;
               AK4_JER_corr_factor = max(0., 1 + JERrand*sqrt(max(pow(sJER,2)-1,0.)));   //want to make sure this is truncated at 0
            }
            if(_verbose)std::cout << "finished with JER" << std::endl;
            AK4_sf_total*= AK4_JER_corr_factor;

         }

      }


      // create corrected (scaled) jet object that will be used for cuts 
      pat::Jet corrJet(*iJet);

      LorentzVector corrJetP4(AK4_sf_total*iJet->px(),AK4_sf_total*iJet->py(),AK4_sf_total*iJet->pz(),AK4_sf_total*iJet->energy());
      corrJet.setP4(corrJetP4);
      if (_verbose)std::cout << "The PU id bool is " << bool(corrJet.userInt("pileupJetIdUpdated:fullId") & (1 << 1) )<< std::endl;

      //measure event HT
      if((corrJet.pt() > 30.)&&(abs(corrJet.eta()) < 2.5)  )totHT+= abs(corrJet.pt() );

      // apply AK4 jet selection (post JEC and JER)
      bool PUID = true;  // assumed true if not applying this
      if(doPUID)
      {
         PUID = false;
         PUID = bool( (corrJet.userInt("pileupJetIdUpdated:fullId") & (1 << 1)) || (corrJet.pt() > 50.) );
      }

      if (  ( corrJet.pt()  < 30. ) ||  (!(corrJet.isPFJet()))  ||  (!isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),PUID, corrJet.pt() ))   ) continue;

      //if( isHEM(corrJet.eta(), corrJet.phi()))return;  //RETURN CUT - if a jet is in the HEM (=bad) region, don't use this event

      double deepJetScore = corrJet.bDiscriminator("pfDeepFlavourJetTags:probb") + corrJet.bDiscriminator("pfDeepFlavourJetTags:probbb")+ corrJet.bDiscriminator("pfDeepFlavourJetTags:problepb");

      if(_verbose)std::cout << "the year is " << year << " with working point value of " << deepjet_wp_tight << ". The hadronFlavour is " << corrJet.hadronFlavour() <<  " and deepjet score " << deepJetScore<< std::endl;
      


      //////////////////////////////////////////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////// b tagging event weight stuff //////////////////////////////////////////
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////
      if(doBtagSF)
      {
         // calculate the b-tag scale factor for the event using the efficiency maps that were loaded in the constructor and correctionlib
         if ( ( runType.find("MC") != std::string::npos  ) || (runType.find("Suu")!= std::string::npos ) )    //only run this on MC (we won't have the hadron flavor stuff for data )
         {

            //these are the bins of the eff histogram you need to draw from
            // histogram size:  63, 0, 6000, 18, -2.4, 2.4   

            // should import the histogram size here so this dosn't have to be manually changed
            int xbin = (int)(corrJet.pt()*bTagEffMap_nPtBins/bTagEffMap_PtRange) + 1;
            int ybin = (int)((corrJet.eta()-bTagEffMap_Eta_low)*bTagEffMap_nEtaBins/(bTagEffMap_Eta_high-bTagEffMap_Eta_low)) +1;

            if(debug) std::cout << "bTagEffMap_nPtBins/bTagEffMap_PtRange/bTagEffMap_Eta_low/bTagEffMap_nEtaBins/bTagEffMap_Eta_high:" <<bTagEffMap_nPtBins << "/"  <<bTagEffMap_PtRange << "/" <<bTagEffMap_Eta_low << "/" <<bTagEffMap_nEtaBins << "/" <<bTagEffMap_Eta_high << std::endl;
            double SF;
            double SF_med;
            double bTag_eff_value, bTag_eff_value_med;

            if(corrJet.hadronFlavour() == 0)   //light jets
            {

               bTag_eff_value = lightjet_eff->GetBinContent(xbin,ybin);
               bTag_eff_value_med = lightjet_eff_med->GetBinContent(xbin,ybin);

               SF = cset_corrector_light->evaluate({"central", "T", 0, std::abs(corrJet.eta()), corrJet.pt()}); /// tight WP SF
               SF_med = cset_corrector_light->evaluate({"central", "M", 0, std::abs(corrJet.eta()), corrJet.pt()});

               //////// MED WP SCALE FACTOR ////////
               if( deepJetScore > deepJet_wp_med)
               {
                  // med WP
                  MC_tagged_med *= bTag_eff_value_med;
                  data_tagged_med *= SF_med*bTag_eff_value_med;

                  if(_verbose)std::cout << "tagged light jet: MC_tagged/data_tagged: " << bTag_eff_value << ":" << SF*bTag_eff_value << std::endl;
               }
               else
               {
                  // med WP
                  MC_notTagged_med *= (1 - bTag_eff_value_med);
                  data_notTagged_med *= (1 - SF_med*bTag_eff_value_med);
               }

               //////// TIGHT WP SCALE FACTOR ///////
               if(_verbose)std::cout << "Scale factor is " << SF << std::endl;
               if(deepJetScore > deepjet_wp_tight)
               {

                  // tight WP
                  MC_tagged *= bTag_eff_value;
                  data_tagged *= SF*bTag_eff_value;
               }

               else
               {
                  // tight WP
                  MC_notTagged *= (1 - bTag_eff_value);
                  data_notTagged *= (1 - SF*bTag_eff_value);

                  if(_verbose)std::cout << "untagged light jet: MC_notTagged/data_notTagged: " << (1- bTag_eff_value) << ":" << (1- SF*bTag_eff_value) << std::endl;
               }
            }
            else if(corrJet.hadronFlavour() == 4) //charm jets
            {
               bTag_eff_value = truecjet_eff->GetBinContent(xbin,ybin);
               bTag_eff_value_med = truecjet_eff_med->GetBinContent(xbin,ybin);

               SF = cset_corrector_bc->evaluate(   {"central", "T", 4, std::abs(corrJet.eta()), corrJet.pt()});
               SF_med = cset_corrector_bc->evaluate(   {"central", "M", 4, std::abs(corrJet.eta()), corrJet.pt()});

               if(_verbose)std::cout << "Scale factor is " << SF << std::endl;

               //////// MED WP SCALE FACTOR ////////
               if(deepJetScore > deepJet_wp_med)
               {
                  // med WP
                  MC_tagged_med  *= bTag_eff_value_med;
                  data_tagged_med *= SF_med*bTag_eff_value_med;
               }
               else
               {
                  // med WP
                  MC_notTagged_med *= (1 - bTag_eff_value_med);
                  data_notTagged_med *= (1 - SF_med*bTag_eff_value_med);

               }

               //////// TIGHT WP SCALE FACTOR ////////
               if(deepJetScore > deepjet_wp_tight)
               {
                  // tight WP
                  MC_tagged  *= bTag_eff_value;
                  data_tagged *= SF*bTag_eff_value;

                  if(_verbose)std::cout << "tagged c jet: MC_tagged/data_tagged: " << bTag_eff_value << ":" << SF*bTag_eff_value << std::endl;
               }
               else
               {
                  // tight WP
                  MC_notTagged *= (1 - bTag_eff_value);
                  data_notTagged *= (1 - SF*bTag_eff_value);

                  if(_verbose)std::cout << "untagged c jet: MC_notTagged/data_notTagged: " << (1- bTag_eff_value) << ":" << (1- SF*bTag_eff_value) << std::endl;
               }
            }
            else if(corrJet.hadronFlavour() == 5) // b jets
            {
               bTag_eff_value = truebjet_eff->GetBinContent(xbin,ybin);
               bTag_eff_value_med = truebjet_eff_med->GetBinContent(xbin,ybin);

               SF             = cset_corrector_bc->evaluate(   {"central", "T", 5, std::abs(corrJet.eta()), corrJet.pt()}); // tight WP
               SF_med         = cset_corrector_bc->evaluate(   {"central", "M", 5, std::abs(corrJet.eta()), corrJet.pt()}); // med WP

               if(_verbose)std::cout << "Scale factor is " << SF << std::endl;

               //////// MED WP SCALE FACTOR ////////
               if(deepJetScore > deepJet_wp_med)
               {
                  // med WP
                  MC_tagged_med  *= bTag_eff_value_med;
                  data_tagged_med *= SF_med*bTag_eff_value_med;
               }
               else
               {
                  // med WP
                  MC_notTagged_med *= (1 - bTag_eff_value_med);
                  data_notTagged_med *= (1 - SF_med*bTag_eff_value_med);
               }

               //////// TIGHT WP SCALE FACTOR ////////
               if(deepJetScore > deepjet_wp_tight)
               {
                  // tight WP
                  MC_tagged  *= bTag_eff_value;
                  data_tagged *= SF*bTag_eff_value;

                  if(_verbose)std::cout << "tagged b jet: MC_tagged/data_tagged: " << bTag_eff_value << ":" << SF*bTag_eff_value << std::endl;
               }
               else
               {  
                  // tight WP
                  MC_notTagged *= (1 - bTag_eff_value);
                  data_notTagged *= (1 - SF*bTag_eff_value);

                  if(_verbose)std::cout << "untagged b jet: MC_notTagged/data_notTagged: " << (1- bTag_eff_value) << ":" << (1- SF*bTag_eff_value) << std::endl;
               }
            }
            else
            {
               std::cout << "ERROR: invalid AK4 hadron flavour. " << std::endl;
            }
            double epsilon = 1e-12;
            if ( (abs(MC_tagged)<epsilon )||(abs(MC_tagged)<epsilon )||(abs(MC_tagged)<epsilon )||(abs(MC_tagged)<epsilon ))
            {
                  if(_verbose)std::cout << "bTag_eff_value/SF/MC_tagged/MC_notTagged/data_tagged/data_notTagged: " << bTag_eff_value<< " / " << SF<< " / " << MC_tagged<< " / " << MC_notTagged << " / " <<data_tagged << " / " << data_notTagged << std::endl;
            }
         }
      }

      int vetoMap_xbin = (int)((corrJet.eta() - jetVetoMap_Xmin)*jetVetoMap_nBinsX/jetVetoMap_XRange ) +1;
      int vetoMap_ybin = (int)((corrJet.phi() - jetVetoMap_Ymin)*jetVetoMap_nBinsY/jetVetoMap_YRange ) +1;       // pt(b) = b*(range / # bins) + b0 -> (pt(b) - b0)*(# bins / range)
      AK4_fails_veto_map[nAK4] = false;
      if(jetVetoMap->GetBinContent(vetoMap_xbin,vetoMap_ybin) > 0)
      {
         AK4_fails_veto_map[nAK4] = true;
      }
       if(isHEM(corrJet.eta(),corrJet.phi()))
      {
         jet_isHEM[nfatjets] = true;
      } 
      else{jet_isHEM[nfatjets] = false;}

      lab_AK4_pt[nAK4] = corrJet.pt();
      AK4_mass[nAK4] = corrJet.mass();
      AK4_eta[nAK4] = corrJet.eta();
      AK4_phi[nAK4] = corrJet.phi();

      AK4_DeepJet_disc[nAK4] = deepJetScore;


      if(nAK4 < 4)
      {
         leadAK4Jets.push_back(TLorentzVector(corrJet.px(),corrJet.py(),corrJet.pz(),corrJet.energy()));
      }


      nAK4++;

   }


   //if(nAK4 <4)return;   // RETURN cut




   ///////////////////// calculate b tag event weights //////////////////////
   if(doBtagSF)
   {
      // tight WP
      bTag_eventWeight_T_nom =  (data_tagged*data_notTagged) / (MC_tagged*MC_notTagged);
      // med WP
      bTag_eventWeight_M_nom =  (data_tagged_med*data_notTagged_med) / (MC_tagged_med*MC_notTagged_med);

      if ((bTag_eventWeight_T_nom != bTag_eventWeight_T_nom) || (std::isinf(bTag_eventWeight_T_nom)) || (bTag_eventWeight_T_nom < 1e-9))
      {
         //std::cout << "BAD BTAG SF: " << bTag_eventWeight_T_nom << std::endl;
         if(_verbose)std::cout << "data_tagged_up/data_notTagged_up/MC_tagged/MC_notTagged: " <<data_tagged << "/" <<data_notTagged << "/" << MC_notTagged<< "/" << MC_notTagged<<  std::endl;
      } 
   }

   lab_nAK4 = nAK4;


   /////////////////////////////////
   // (3) Apply loose HT selection criteria  (= looser than grid points)
   /////////////////////////////////


    if( totHT < 1300) return; // normal HT cut



   ///////////////////////////////////
   ////// Calculate dijet masses /////
   ///////////////////////////////////



   /*
   // calculate the candidate dijet delta R values
   double minDeltaRDisc12 = sqrt( pow(leadAK4Jets[0].DeltaR(leadAK4Jets[1]),2) + pow(leadAK4Jets[2].DeltaR(leadAK4Jets[3]),2));    // dijet one always has j1 in it
   double minDeltaRDisc13 = sqrt( pow(leadAK4Jets[0].DeltaR(leadAK4Jets[2]),2) + pow(leadAK4Jets[1].DeltaR(leadAK4Jets[3]),2));
   double minDeltaRDisc14 = sqrt( pow(leadAK4Jets[0].DeltaR(leadAK4Jets[3]),2) + pow(leadAK4Jets[1].DeltaR(leadAK4Jets[2]),2));

   if (  abs(min(minDeltaRDisc12, min(minDeltaRDisc13,minDeltaRDisc14)) -minDeltaRDisc12)<1e-8 ) 
   {
      //set dijet masses
      dijetMassOne = (leadAK4Jets[0] +leadAK4Jets[1]).M();
      dijetMassTwo = (leadAK4Jets[2] +leadAK4Jets[3]).M();
   }
   else if (  abs(min(minDeltaRDisc12, min(minDeltaRDisc13,minDeltaRDisc14)) -minDeltaRDisc13)<1e-8 ) 
   {
      // set dijet masses
      dijetMassOne = (leadAK4Jets[0] +leadAK4Jets[2]).M();
      dijetMassTwo = (leadAK4Jets[1] +leadAK4Jets[3]).M();
   }
   else if (  abs(min(minDeltaRDisc12, min(minDeltaRDisc13,minDeltaRDisc14)) -minDeltaRDisc14)<1e-8 ) 
   {
      //set dijet masses
      dijetMassOne = (leadAK4Jets[0] +leadAK4Jets[3]).M();
      dijetMassTwo = (leadAK4Jets[1] +leadAK4Jets[2]).M();
   }

   */

   if(_verbose)std::cout << "before AK8 jets" << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////_AK8 Jets_/////////////////////////////////////////////////////////////////////
   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   edm::Handle<std::vector<pat::Jet> > fatJets;
   iEvent.getByToken(fatJetToken_, fatJets);
   std::vector<reco::LeafCandidate> candsBoosted;   
   std::vector<reco::LeafCandidate> candsUnboosted;
   std::vector<fastjet::PseudoJet> superJetOne;        //jets for dot prduct #cos(theta) > 0
   std::vector<fastjet::PseudoJet> superJetTwo;
   std::vector<TLorentzVector> superJetOneLV;        //workaround for weird fastjet bug
   std::vector<TLorentzVector> superJetTwoLV; 
   std::vector<TLorentzVector> selectedAK8_TLV;

   JME::JetResolution resolution = JME::JetResolution::get(iSetup, "AK8PF_pt");                          //load JER stuff from global tag
   JME::JetResolutionScaleFactor resolution_sf = JME::JetResolutionScaleFactor::get(iSetup, "AK8PF");

   nfatjets = 0;
   nfatjet_pre = 0;

   TLorentzVector allAK8(0,0,0,0);



   /////////// (4) set grid variable variable counters to zero (ex. nAK8_Et250 = 0;) ///////////
   nAK8_ET100 = 0; nAK8_ET150 = 0; nAK8_ET200 = 0; nAK8_ET250 = 0; nAK8_ET300 = 0; nAK8_ET350 = 0; nAK8_ET400 = 0; 
   nHeavyAK8_pt600_M10 = 0; nHeavyAK8_pt600_M20 = 0; nHeavyAK8_pt600_M30 = 0; nHeavyAK8_pt600_M40 = 0; nHeavyAK8_pt600_M45 = 0; nHeavyAK8_pt600_M50 = 0; nHeavyAK8_pt600_M55 = 0; nHeavyAK8_pt600_M60 = 0; nHeavyAK8_pt600_M70 = 0; 
   nHeavyAK8_pt500_M10 = 0; nHeavyAK8_pt500_M20 = 0; nHeavyAK8_pt500_M30 = 0; nHeavyAK8_pt500_M40 = 0; nHeavyAK8_pt500_M45 = 0; nHeavyAK8_pt500_M50 = 0; nHeavyAK8_pt500_M55 = 0; nHeavyAK8_pt500_M60 = 0; nHeavyAK8_pt500_M70 = 0; 
   nHeavyAK8_pt400_M10 = 0; nHeavyAK8_pt400_M20 = 0; nHeavyAK8_pt400_M30 = 0; nHeavyAK8_pt400_M40 = 0; nHeavyAK8_pt400_M45 = 0; nHeavyAK8_pt400_M50 = 0; nHeavyAK8_pt400_M55 = 0; nHeavyAK8_pt400_M60 = 0; nHeavyAK8_pt400_M70 = 0; 
   nHeavyAK8_pt300_M10 = 0; nHeavyAK8_pt300_M20 = 0; nHeavyAK8_pt300_M30 = 0; nHeavyAK8_pt300_M40 = 0; nHeavyAK8_pt300_M45 = 0; nHeavyAK8_pt300_M50 = 0; nHeavyAK8_pt300_M55 = 0; nHeavyAK8_pt300_M60 = 0; nHeavyAK8_pt300_M70 = 0; 
   nHeavyAK8_pt200_M10 = 0; nHeavyAK8_pt200_M20 = 0; nHeavyAK8_pt200_M30 = 0; nHeavyAK8_pt200_M40 = 0; nHeavyAK8_pt200_M45 = 0; nHeavyAK8_pt200_M50 = 0; nHeavyAK8_pt200_M55 = 0; nHeavyAK8_pt200_M60 = 0; nHeavyAK8_pt200_M70 = 0; 


   /////////////////////////////////////////////////////////////////////////////////////////////


   double tot_jet_px=0,tot_jet_py=0,tot_jet_pz=0, tot_jet_E=0;
   std::vector<TLorentzVector> leadAK8Jets;


   // loop over AK8 jets, save information for event selection, grab particles to create superjets
   for(auto iJet = fatJets->begin(); iJet != fatJets->end(); iJet++)    
   {


      if( (sqrt(pow(iJet->mass(),2)+pow(iJet->pt(),2)) < 25.) || (!(iJet->isPFJet())) || ( abs(iJet->eta()) > 2.4 )) continue;   //don't even bother with these jets

      double AK8_sf_total = 1.0;     // this scales jet/particle 4-vectors, compounds all scale factors

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //////////   _JET ENERGY RESOLUTION STUFF //////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////
      if(doJER)
      {
         double AK8_JER_corr_factor = 1.0; // this won't be touched for data
         
         if( (runType.find("MC") != std::string::npos)|| (runType.find("Suu")!= std::string::npos ) ) //if((runType == "SigMC") || (runType == "QCDMC") || (runType == "TTbarMC") ) 
         {
            if(_verbose)std::cout << "doing JER" << std::endl;

            double sJER     = -9999.;    //JER scale factor
            double sigmaJER = -9999.;    //this is the "resolution" you get from the scale factors 
            
            //these are for getting the JER scale factors
            JME::JetParameters parameters_1;
            parameters_1.setJetPt(iJet->pt());
            parameters_1.setJetEta(iJet->eta());
            parameters_1.setRho(*rho);
            sigmaJER = resolution.getResolution(parameters_1);   //pT resolution

            //JME::JetParameters
            JME::JetParameters parameters;
            parameters.setJetPt(iJet->pt());
            parameters.setJetEta(iJet->eta());
            sJER =  resolution_sf.getScaleFactor(parameters  );  //{{JME::Binning::JetEta, iJet->eta()}});

            const reco::GenJet *genJet = iJet->genJet();
            if( genJet)   // try the first technique
            {
               AK8_JER_corr_factor = 1 + (sJER - 1)*(iJet->pt()-genJet->pt())/iJet->pt();
            }
            else   // if no gen jet is matched, try the second technique
            {
               randomNum->SetSeed( abs(static_cast<int>(iJet->phi()*1e4)) );
               double JERrand = randomNum->Gaus(0.0, sigmaJER);
               //double JERrand = 1.0 + sigmaJER;
               AK8_JER_corr_factor = max(0., 1 + JERrand*sqrt(max(pow(sJER,2)-1,0.)));   //want to make sure this is truncated at 0
            }

            if(_verbose)std::cout << "finished with JER" << std::endl;
         }
         AK8_sf_total*= AK8_JER_corr_factor;
      }

      /////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////

      // create corrected (scaled) jet object that will be used for cuts 
      pat::Jet corrJet(*iJet);
      LorentzVector corrJetP4(AK8_sf_total*iJet->px(),AK8_sf_total*iJet->py(),AK8_sf_total*iJet->pz(),AK8_sf_total*iJet->energy());
      corrJet.setP4(corrJetP4);

      if(_verbose)  std::cout << "nominal p4: " << iJet->px()<< "," <<iJet->py() << "," << iJet->pz()<< "," << iJet->energy()<< std::endl;
      if(_verbose)  std::cout << "corrected p4: " << corrJet.px()<< "," <<corrJet.py() << "," << corrJet.pz()<< "," << corrJet.energy()<< std::endl;

      if((corrJet.pt() > 500.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 45.)) 
      {
         if(isHEM(corrJet.eta(),corrJet.phi())) jet_pre_isHEM[nfatjet_pre] = true;
         else{ jet_pre_isHEM[nfatjet_pre] = false;}

         nfatjet_pre++;
      }

      /////////// (5) calculate grid variables here ///////////
      if((sqrt(pow(corrJet.mass(),2)+pow(corrJet.pt(),2)) > 100.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets )) && (corrJet.mass()> 0.)) nAK8_ET100++;
      if((sqrt(pow(corrJet.mass(),2)+pow(corrJet.pt(),2)) > 150.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets )) && (corrJet.mass()> 0.)) nAK8_ET150++;
      if((sqrt(pow(corrJet.mass(),2)+pow(corrJet.pt(),2)) > 200.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets )) && (corrJet.mass()> 0.)) nAK8_ET200++;
      if((sqrt(pow(corrJet.mass(),2)+pow(corrJet.pt(),2)) > 250.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets )) && (corrJet.mass()> 0.)) nAK8_ET250++;
      if((sqrt(pow(corrJet.mass(),2)+pow(corrJet.pt(),2)) > 300.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets )) && (corrJet.mass()> 0.)) nAK8_ET300++;
      if((sqrt(pow(corrJet.mass(),2)+pow(corrJet.pt(),2)) > 350.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets )) && (corrJet.mass()> 0.)) nAK8_ET350++;
      if((sqrt(pow(corrJet.mass(),2)+pow(corrJet.pt(),2)) > 400.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets )) && (corrJet.mass()> 0.)) nAK8_ET400++;

      if((corrJet.pt() > 600.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 10.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt600_M10++;
      if((corrJet.pt() > 600.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 20.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt600_M20++;
      if((corrJet.pt() > 600.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 30.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt600_M30++;
      if((corrJet.pt() > 600.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 40.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt600_M40++;
      if((corrJet.pt() > 600.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 45.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt600_M45++;
      if((corrJet.pt() > 600.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 50.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt600_M50++;
      if((corrJet.pt() > 600.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 55.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt600_M55++;
      if((corrJet.pt() > 600.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 60.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt600_M60++;
      if((corrJet.pt() > 600.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 70.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt600_M70++;
      
      if((corrJet.pt() > 500.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 10.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt500_M10++;
      if((corrJet.pt() > 500.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 20.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt500_M20++;
      if((corrJet.pt() > 500.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 30.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt500_M30++;
      if((corrJet.pt() > 500.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 40.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt500_M40++;
      if((corrJet.pt() > 500.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 45.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt500_M45++;
      if((corrJet.pt() > 500.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 50.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt500_M50++;
      if((corrJet.pt() > 500.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 55.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt500_M55++;
      if((corrJet.pt() > 500.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 60.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt500_M60++;
      if((corrJet.pt() > 500.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 70.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt500_M70++;
      

      if((corrJet.pt() > 400.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 10.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt400_M10++;
      if((corrJet.pt() > 400.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 20.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt400_M20++;
      if((corrJet.pt() > 400.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 30.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt400_M30++;
      if((corrJet.pt() > 400.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 40.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt400_M40++;
      if((corrJet.pt() > 400.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 45.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt400_M45++;
      if((corrJet.pt() > 400.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 50.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt400_M50++;
      if((corrJet.pt() > 400.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 55.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt400_M55++;
      if((corrJet.pt() > 400.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 60.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt400_M60++;
      if((corrJet.pt() > 400.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 70.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt400_M70++;

      if((corrJet.pt() > 300.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 10.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt300_M10++;
      if((corrJet.pt() > 300.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 20.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt300_M20++;
      if((corrJet.pt() > 300.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 30.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt300_M30++;
      if((corrJet.pt() > 300.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 40.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt300_M40++;
      if((corrJet.pt() > 300.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 45.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt300_M45++;
      if((corrJet.pt() > 300.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 50.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt300_M50++;
      if((corrJet.pt() > 300.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 55.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt300_M55++;
      if((corrJet.pt() > 300.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 60.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt300_M60++;
      if((corrJet.pt() > 300.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 70.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt300_M70++;

      if((corrJet.pt() > 200.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 10.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt200_M10++;
      if((corrJet.pt() > 200.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 20.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt200_M20++;
      if((corrJet.pt() > 200.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 30.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt200_M30++;
      if((corrJet.pt() > 200.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 40.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt200_M40++;
      if((corrJet.pt() > 200.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 45.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt200_M45++;
      if((corrJet.pt() > 200.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 50.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt200_M50++;
      if((corrJet.pt() > 200.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 55.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt200_M55++;
      if((corrJet.pt() > 200.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 60.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt200_M60++;
      if((corrJet.pt() > 200.) && ((corrJet.isPFJet())) && (isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets) ) && (corrJet.userFloat("ak8PFJetsPuppiSoftDropMass") > 70.) && (!isHEM(corrJet.eta(),corrJet.phi()))) nHeavyAK8_pt200_M70++;


      ///////////////////////////////////////////////////////

      if((sqrt(pow(corrJet.mass(),2)+pow(corrJet.pt(),2)) < 200.) || (!(corrJet.isPFJet())) || (!isgoodjet(corrJet.eta(),corrJet.neutralHadronEnergyFraction(), corrJet.neutralEmEnergyFraction(),corrJet.numberOfDaughters(),corrJet.chargedHadronEnergyFraction(),corrJet.chargedMultiplicity(),corrJet.muonEnergyFraction(),corrJet.chargedEmEnergyFraction(),nfatjets )) || (corrJet.mass()< 0.)) continue; //userFloat("ak8PFJetsPuppiSoftDropMass")
      
      if(isHEM(corrJet.eta(),corrJet.phi()))
      {
         fatjet_isHEM[nfatjets] = true;
      } 
      else{fatjet_isHEM[nfatjets] = false;}


      if(isHEM(corrJet.eta(),corrJet.phi()))
      {
         jet_isHEM[nfatjets] = true;
      } 
      else{jet_isHEM[nfatjets] = false;}

      if(nfatjets < 4)
      {
         leadAK8Jets.push_back(TLorentzVector(corrJet.px(),corrJet.py(),corrJet.pz(),corrJet.energy()));
      }

      int vetoMap_xbin = (int)((corrJet.eta() - jetVetoMap_Xmin)*jetVetoMap_nBinsX/jetVetoMap_XRange ) +1;
      int vetoMap_ybin = (int)((corrJet.phi() - jetVetoMap_Ymin)*jetVetoMap_nBinsY/jetVetoMap_YRange ) +1;       // pt(b) = b*(range / # bins) + b0 -> (pt(b) - b0)*(# bins / range)
      AK8_fails_veto_map[nfatjets] = false;
      if(jetVetoMap->GetBinContent(vetoMap_xbin,vetoMap_ybin) > 0.)
      {
         AK8_fails_veto_map[nfatjets] = true;
      }
      jet_pt[nfatjets] = corrJet.pt();
      jet_phi[nfatjets] = iJet->phi();  
      jet_eta[nfatjets] = corrJet.eta();
      jet_mass[nfatjets] = corrJet.mass();
      for (unsigned int iii=0; iii<iJet->numberOfDaughters();iii++)   // get all jet particles
      {
         const reco::Candidate* iJ = iJet->daughter(iii);
         const pat::PackedCandidate* candJetbegin = (pat::PackedCandidate*) iJ;
         double puppiweight = candJetbegin->puppiWeight();
         candsUnboosted.push_back(LeafCandidate(iJet->daughter(iii)->charge(), Particle::LorentzVector(AK8_sf_total*puppiweight*iJ->px(), AK8_sf_total*puppiweight*iJ->py(), AK8_sf_total*puppiweight*iJ->pz(), AK8_sf_total*puppiweight*iJ->energy())));
         tot_jet_px+=AK8_sf_total*puppiweight*candJetbegin->px();tot_jet_py+=AK8_sf_total*puppiweight*candJetbegin->py();tot_jet_pz+=AK8_sf_total*puppiweight*candJetbegin->pz();tot_jet_E+=AK8_sf_total*puppiweight*candJetbegin->energy();
      }

      nfatjets++;
   }
   
   diAK8Jet_mass[0] = 0; diAK8Jet_mass[1] = 0;
   fourAK8JetMass = 0;
   if(nfatjets >3)
   {

      fourAK8JetMass = (leadAK8Jets[0] + leadAK8Jets[1] +  leadAK8Jets[2] + leadAK8Jets[3]).M();
      double minDeltaRDisc12_AK8 = sqrt( pow(leadAK8Jets[0].DeltaR(leadAK8Jets[1]),2) + pow(leadAK8Jets[2].DeltaR(leadAK8Jets[3]),2));    // dijet one always has j1 in it
      double minDeltaRDisc13_AK8 = sqrt( pow(leadAK8Jets[0].DeltaR(leadAK8Jets[2]),2) + pow(leadAK8Jets[1].DeltaR(leadAK8Jets[3]),2));
      double minDeltaRDisc14_AK8 = sqrt( pow(leadAK8Jets[0].DeltaR(leadAK8Jets[3]),2) + pow(leadAK8Jets[1].DeltaR(leadAK8Jets[2]),2));

      if (  abs(min(minDeltaRDisc12_AK8, min(minDeltaRDisc13_AK8,minDeltaRDisc14_AK8)) - minDeltaRDisc12_AK8)<1e-8 ) 
      {
         //set dijet masses
         diAK8Jet_mass[0] = (leadAK8Jets[0] +leadAK8Jets[1]).M();
         diAK8Jet_mass[1] = (leadAK8Jets[2] +leadAK8Jets[3]).M();
      }
      else if (  abs(min(minDeltaRDisc12_AK8, min(minDeltaRDisc13_AK8,minDeltaRDisc14_AK8)) -minDeltaRDisc13_AK8)<1e-8 ) 
      {
         // set dijet masses
         diAK8Jet_mass[0] = (leadAK8Jets[0] +leadAK8Jets[2]).M();
         diAK8Jet_mass[1] = (leadAK8Jets[1] +leadAK8Jets[3]).M();
      }
      else if (  abs(min(minDeltaRDisc12_AK8, min(minDeltaRDisc13_AK8,minDeltaRDisc14_AK8)) -minDeltaRDisc14_AK8)<1e-8 ) 
      {
         //set dijet masses
         diAK8Jet_mass[0] = (leadAK8Jets[0] +leadAK8Jets[3]).M();
         diAK8Jet_mass[1] = (leadAK8Jets[1] +leadAK8Jets[2]).M();
      }
   }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////////////////////////////////////////////_Clustering_///////////////////////////////////////////////////////////////////
   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //     This is the point where superjets will be formed !

   if(_verbose)std::cout << "before cluster" << std::endl;
   /////calculate COM frame boost////////////////////////////////////////////////////////////////////?????///////////////
   TVector3 totJetBeta = TVector3(tot_jet_px/tot_jet_E,tot_jet_py/tot_jet_E ,tot_jet_pz/tot_jet_E);


   //////////boost all jet particles into MPP frame//////////////////////////////////////////////////////////////////////
   std::vector<TLorentzVector> candsBoostedTLV_;
   double pxCOM = 0,pyCOM=0,pzCOM=0;
   for(auto iC = candsUnboosted.begin();iC != candsUnboosted.end(); iC++)
   {
      TLorentzVector iC_(iC->px(),iC->py(),iC->pz(),iC->energy());
      iC_.Boost(-totJetBeta.X(),-totJetBeta.Y(),-totJetBeta.Z());
      pxCOM+=iC_.Px();pyCOM+=iC_.Py();pzCOM+=iC_.Pz();
      candsBoostedTLV_.push_back( iC_ );

   }
   std::vector<fastjet::PseudoJet> candsBoostedPJ;
   for(auto iP=candsBoostedTLV_.begin();iP!=candsBoostedTLV_.end();iP++ )
   {
      candsBoostedPJ.push_back( fastjet::PseudoJet(iP->Px(),iP->Py(),iP->Pz(),iP->E()   ) );
   }

   ///////take all boosted particles in MPP frame and recluster, then sort these AK8 jets
   double R0 = 0.8;

    //fastjet::JetDefinition jet_def0(fastjet::antikt_algorithm, R0);  // alternate jet clustering algorithm
   fastjet::JetDefinition jet_def0(fastjet::cambridge_algorithm, R0);
   fastjet::ClusterSequence cs_jet0(candsBoostedPJ, jet_def0); 

   // shedding particles happens here -> can change this energy threshold, just changed it from 50. 
   std::vector<fastjet::PseudoJet> jetsFJ_jet0 = fastjet::sorted_by_E(cs_jet0.inclusive_jets(10.));
   double MPP_COM2_px = 0,MPP_COM2_py=0,MPP_COM2_pz =0,MPP_COM2_E = 0;
   int counter = 0;
   int nReclustered_CA8 = 0;
   std::vector<TLorentzVector> candsBoostedTLV;   
   for (auto iJet=jetsFJ_jet0.begin(); iJet<jetsFJ_jet0.end(); iJet++)                             
   {   
      if(iJet->E() > 250.) nReclustered_CA8++;
      if (iJet->constituents().size() > 0)
      {
         for(auto iDaughter = iJet->constituents().begin(); iDaughter != iJet->constituents().end(); iDaughter++)
         {
            counter++;
            candsBoosted.push_back(reco::LeafCandidate(+1, reco::Candidate::LorentzVector(iDaughter->px(), iDaughter->py(), iDaughter->pz(), iDaughter->E())));
            candsBoostedTLV.push_back(TLorentzVector( iDaughter->px(), iDaughter->py(), iDaughter->pz(), iDaughter->E()    ));
            if(counter > 300)return; // RETURN cut, event is bad because of fastjet/root bug
         }
      }
      MPP_COM2_px+= iJet->px(); MPP_COM2_py+= iJet->py(); MPP_COM2_pz+=iJet->pz() ; MPP_COM2_E+=iJet->E() ;
   }
   if(_verbose)std::cout << "In this event, there were " << nfatjets << " lab frame AK8 jets selected, and " << nReclustered_CA8 << " reclustered MPP frame CA jets w/ E> 250 GeV" << std::endl;
    TVector3 MPP_COM2 = TVector3(MPP_COM2_px/MPP_COM2_E,MPP_COM2_py/MPP_COM2_E ,MPP_COM2_pz/MPP_COM2_E);
    std::vector<reco::LeafCandidate> candsBoosted_MPPCOM2;   

    for(auto iPart = candsBoostedTLV.begin(); iPart != candsBoostedTLV.end(); iPart++)
    {
        iPart->Boost(-MPP_COM2.X(), -MPP_COM2.Y(), -MPP_COM2.Z());
        candsBoosted_MPPCOM2.push_back( reco::LeafCandidate(+1, reco::Candidate::LorentzVector( iPart->Px(),iPart->Py(),iPart->Pz(),iPart->E()   ) )       );
    }

    Thrust thrust_(candsBoosted.begin(), candsBoosted.end());                             //thrust axis in COM frame
    math::XYZVector thrustAxis = thrust_.axis();
    TVector3 thrust_vector(thrustAxis.X(),thrustAxis.Y(),thrustAxis.Z());
   


///////////////////////////get vector of sorted superjets//////////////////////////////////////////////////////   
    if(_verbose)std::cout << "sorted jets" << std::endl;

    std::vector<TLorentzVector> negSuperJet_preSort;
    std::vector<TLorentzVector> posSuperJet_preSort;
    std::vector<TLorentzVector> miscJets;

    for (auto iJet=jetsFJ_jet0.begin(); iJet<jetsFJ_jet0.end(); iJet++)                             
    {
        TLorentzVector candJet(iJet->px(),iJet->py(),iJet->pz(),iJet->E());
        TVector3 candJet_vec = candJet.Vect();
        double cosAngle = cos(candJet_vec.Angle(thrust_vector));
        if    (cosAngle < -0.85) negSuperJet_preSort.push_back(candJet);  
        else if(cosAngle > 0.85) posSuperJet_preSort.push_back(candJet);
        else{ miscJets.push_back(candJet); }
    }
   if(_verbose)std::cout << "enter sort jets" << std::endl;

   std::vector<TLorentzVector> negSuperJet;
   std::vector<TLorentzVector> posSuperJet;

   if(_verbose)std::cout << "In this event there are " << posSuperJet_preSort.size() << "/" << negSuperJet_preSort.size() << "/" << miscJets.size() << "pos/neg/misc jets" << std::endl;

   if(( miscJets.size() > 0) )
   {   
      sortJets testing(negSuperJet_preSort,posSuperJet_preSort,miscJets);
      negSuperJet = testing.finalSuperJet1;
      posSuperJet = testing.finalSuperJet2;
   }
   else
   {
      negSuperJet = negSuperJet_preSort;
      posSuperJet = posSuperJet_preSort;
   }
   if( ((negSuperJet.size() < 1 ) || (posSuperJet.size()<1) )  ) 
   {
      if(_verbose)std::cout << "Superjet with size < 1: superjet1 size is " << posSuperJet.size() << ", superjet 2 size is "<< negSuperJet.size()  <<"| EXITING ----------------" <<std::endl;
      return; // RETURN cut - event is bad because of one superjet has 0 jets
   }
   TLorentzVector SJ1(0,0,0,0);
   TLorentzVector SJ2(0,0,0,0);

   for(unsigned int iii = 0; iii < negSuperJet.size();iii++)
   {
      SJ2+=negSuperJet[iii];
   }

   for(unsigned int iii = 0; iii < posSuperJet.size();iii++)
   {
      SJ1+=posSuperJet[iii];
   }
///////////////////////////////////////////////////////////////////////////////////////////

   /////////// Associating lab AK8 jets with one of the COM reclustered AK8 jets which gives assignment of lab AK8 jets to the superjets
   int lab_AK8_num = 0;

   for(auto iLabAK8 = selectedAK8_TLV.begin(); iLabAK8!=selectedAK8_TLV.end(); iLabAK8++)
   { 
      //abs(candJet.Angle(iJet->Vect()))
      iLabAK8->Boost(-totJetBeta.X(),-totJetBeta.Y(),-totJetBeta.Z()); // boost jet to COM frame
      double minAngle = 99999;
      AK8_SJ_assignment[lab_AK8_num] = -999;
      for( auto iSJ_jet = negSuperJet.begin(); iSJ_jet!= negSuperJet.end();iSJ_jet++ )
      {
         if( abs(iSJ_jet->Angle(iLabAK8->Vect())) < minAngle)
         {
            AK8_SJ_assignment[lab_AK8_num] = 1; // means NEGATIVE superjet
            minAngle =  abs(iSJ_jet->Angle(iLabAK8->Vect()));
         } 
      }
      for( auto iSJ_jet = posSuperJet.begin(); iSJ_jet!= posSuperJet.end();iSJ_jet++ )
      {
         if( abs(iSJ_jet->Angle(iLabAK8->Vect())) < minAngle)
         {
            AK8_SJ_assignment[lab_AK8_num] = 0; // means POSITIVE superjet
            minAngle =  abs(iSJ_jet->Angle(iLabAK8->Vect()));
         } 
      } 
      lab_AK8_num++;
   } 

   for(int jjj = 0; jjj < nfatjets; jjj++) // loop over all AK8 jets
   {
      for(int iii = 0; iii < nAK4; iii++) // loop over all AK4 jets
      {
         if (lab_AK4_AK8_parent[iii] != -1 ) // if the AK4 jet actually has a matched parent
         {
            if( lab_AK4_AK8_parent[iii] == jjj ) // if the AK4 jet is a daughter of this AK8 jet
            {
               AK4_SJ_assignment[iii] = AK8_SJ_assignment[jjj]; // the AK4 jet gets the AK8 jet superjet assignment
            }
         }
      }
   }

            
   ///// selectedAK8_TLV are now boosted!!

    //get vector of pseudo jet particles for all particles in each superjet, one per superjet
    for (auto iJet=jetsFJ_jet0.begin(); iJet<jetsFJ_jet0.end(); iJet++)                             
    {   
        TLorentzVector candJet(iJet->px(),iJet->py(),iJet->pz(),iJet->E());

        int SJMatch = 0;
        if      (isMatchedtoSJ(posSuperJet,candJet)) SJMatch = 1;
        else if (isMatchedtoSJ(negSuperJet,candJet)) SJMatch = 2;
        else if(SJMatch == 0) std::cout << "Something didn't work... " << std::endl;


        //sort jet particles into either SuperJet 1 or SuperJet 2
        if (iJet->constituents().size() > 0)
        {
            for(auto iJ = iJet->constituents().begin(); iJ != iJet->constituents().end(); iJ++)
            {  
                if     (SJMatch==1)superJetOneLV.push_back(TLorentzVector(iJ->px(),iJ->py(),iJ->pz(),iJ->E()));
                else if(SJMatch==2)superJetTwoLV.push_back(TLorentzVector(iJ->px(),iJ->py(),iJ->pz(),iJ->E()));
            }
         }        
   }

   for(auto iPart = superJetOneLV.begin(); iPart != superJetOneLV.end(); iPart++)
   {
         superJetOne.push_back(fastjet::PseudoJet(iPart->Px(),iPart->Py(),iPart->Pz(),iPart->E()));
   }
   for(auto iPart = superJetTwoLV.begin(); iPart != superJetTwoLV.end(); iPart++)
   {
        superJetTwo.push_back(fastjet::PseudoJet(iPart->Px(),iPart->Py(),iPart->Pz(),iPart->E()));
   }



   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
   ////////////////////////////////////////// superjet analysis /////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////


   if(_verbose)std::cout << "Starting superjet loop." << std::endl;

   double superJetpx,superJetpy,superJetpz,superJetE;
   std::vector<std::vector<fastjet::PseudoJet>> superJets;
   superJets.push_back(superJetOne);
   superJets.push_back(superJetTwo);

   nSuperJets = 0;   
   tot_nAK4_70 = 0; 
   tot_nAK4_50 = 0;
   tot_mpp_AK4 = 0;
   double diSuperJet_E = 0, diSuperJet_px = 0,diSuperJet_py = 0,diSuperJet_pz =0;
   double diSuperJet_E_100 = 0, diSuperJet_px_100 = 0,diSuperJet_py_100 = 0,diSuperJet_pz_100 =0;
   if(_verbose)std::cout << "about to loop over superjets" << std::endl;

   for(auto iSJ = superJets.begin();iSJ!= superJets.end();iSJ++)
   {
     if(_verbose)std::cout << "starting superjet" << nSuperJets <<std::endl;

      ///////////////// /inside superjet loop //////////////////////
      ///////         fill BESTmap for superjet ////////////////////
      //////////////////////////////////////////////////////////////


      //////get BEST scores for superjet /////////
      BESTmap.clear();
      BESTmap["tot_pt"] = allAK8.Pt();
      BESTmap["tot_HT"] = totHT;

      BESTmap["eventNumber"] = eventNumber;


      /*
      std::cout << "-------------------------------- new SJ ----------------------------------------- " << std::endl;
      std::cout << "eventNumber" << " " << eventNumber << std::endl;
      std::cout << "superjet_num" << " " << nSuperJets + 1<< std::endl;
      std::cout << "tot_HT" << " " << totHT << std::endl;
      //std::cout << "tot_pt" << " " << allAK8.Pt() << std::endl;
      */

      if(_verbose)std::cout << "Filling BEST map in SJ " << nSuperJets <<std::endl;
      std::vector<float> BESTScores;
      
      if (!fillSJVars(BESTmap, *iSJ,nSuperJets))    //if this fails somehow, event is skipped
      {
         return;   // RETURN cut - tree was filled incorrectly
      }
      if(_verbose)std::cout << "Filled BEST map. Now getting prediction in SJ " << nSuperJets <<std::endl;
      

      BESTScores = BEST_->getPrediction(BESTmap);
      
      ///store BEST scores in tree
      int decision = std::distance(BESTScores.begin(), std::max_element(BESTScores.begin(), BESTScores.end() ) );

      if(_verbose)std::cout << "Got BEST prediction and decision in SJ " << nSuperJets <<std::endl;
      
      if (nSuperJets == 0)
      {
         SJ1_BEST_scores[0] = (double) BESTScores[0];
         SJ1_BEST_scores[1] = (double) BESTScores[1];
         SJ1_BEST_scores[2] = (double) BESTScores[2];
         SJ1_decision = decision;
         std::cout << "SJ1_decision: " << SJ1_decision << std::endl;
      }
      else if (nSuperJets == 1)
      {
         SJ2_BEST_scores[0] = (double) BESTScores[0];
         SJ2_BEST_scores[1] = (double) BESTScores[1];
         SJ2_BEST_scores[2] = (double) BESTScores[2];
         SJ2_decision = decision;
         std::cout << "SJ2_decision: " << SJ2_decision << std::endl;
      } 
   
      

      superJetpx=0;superJetpy=0;superJetpz=0;superJetE=0;
      for(auto iP = iSJ->begin();iP != iSJ->end();iP++)
      {
         superJetpx+=iP->px();superJetpy+=iP->py();superJetpz+=iP->pz();superJetE +=iP->E();
         diSuperJet_E +=iP->E();diSuperJet_px+=iP->px();diSuperJet_py+=iP->py();diSuperJet_pz+=iP->pz();
      }
      superJet_mass[nSuperJets] = sqrt(pow(superJetE,2)-pow(superJetpx,2)-pow(superJetpy,2)-pow(superJetpz,2));
      TLorentzVector superJetTLV(superJetpx,superJetpy,superJetpz,superJetE);    //Lorentz vector representing jet axis -> now minimize the parallel momentum

      std::vector<fastjet::PseudoJet> boostedSuperJetPart;

      //boost particles in SuperJet to its COM frame
      for(auto iP = iSJ->begin();iP != iSJ->end();iP++)
      {
         TLorentzVector iP_(iP->px(),iP->py(),iP->pz(),iP->E());
         iP_.Boost(-superJetpx/superJetE,-superJetpy/superJetE,-superJetpz/superJetE);
         boostedSuperJetPart.push_back(fastjet::PseudoJet(iP_.Px(),iP_.Py(),iP_.Pz(),iP_.E()));
      }

      ///reclustering SuperJet that is now boosted into the MPP frame
      double R = 0.4;
      //fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, R);
      fastjet::JetDefinition jet_def(fastjet::cambridge_algorithm, R);
      fastjet::ClusterSequence cs_jet(boostedSuperJetPart, jet_def); 
      std::vector<fastjet::PseudoJet> jetsFJ_jet = fastjet::sorted_by_E(cs_jet.inclusive_jets(0.0));

      double SJ_50_px = 0, SJ_50_py=0,SJ_50_pz=0,SJ_50_E=0;
      double SJ_70_px = 0, SJ_70_py=0,SJ_70_pz=0,SJ_70_E=0;
      double SJ_100_px = 0, SJ_100_py=0,SJ_100_pz=0,SJ_100_E=0;
      double SJ_125_px = 0, SJ_125_py=0,SJ_125_pz=0,SJ_125_E=0;
      double SJ_150_px = 0, SJ_150_py=0,SJ_150_pz=0,SJ_150_E=0;
      double SJ_200_px = 0, SJ_200_py=0,SJ_200_pz=0,SJ_200_E=0;
      double SJ_300_px = 0, SJ_300_py=0,SJ_300_pz=0,SJ_300_E=0;
      double SJ_400_px = 0, SJ_400_py=0,SJ_400_pz=0,SJ_400_E=0;
      double SJ_600_px = 0, SJ_600_py=0,SJ_600_pz=0,SJ_600_E=0;
      double SJ_800_px = 0, SJ_800_py=0,SJ_800_pz=0,SJ_800_E=0;
      double SJ_1000_px = 0, SJ_1000_py=0,SJ_1000_pz=0,SJ_1000_E=0;

      SJ_nAK4_50[nSuperJets] = 0,  SJ_nAK4_70[nSuperJets] =0,SJ_nAK4_100[nSuperJets] = 0,SJ_nAK4_125[nSuperJets] = 0,SJ_nAK4_150[nSuperJets] = 0,SJ_nAK4_200[nSuperJets] = 0,SJ_nAK4_300[nSuperJets] = 0,SJ_nAK4_400[nSuperJets] = 0,SJ_nAK4_1000[nSuperJets] = 0;
      SJ_nAK4_600[nSuperJets] = 0, SJ_nAK4_800[nSuperJets] = 0;

      /* not using these vars now, but could in the future
         double SJ1_AK41_px, SJ1_AK41_py, SJ1_AK41_pz, SJ1_AK41_E;
         double SJ1_AK42_px, SJ1_AK42_py, SJ1_AK42_pz, SJ1_AK42_E;
         double SJ1_AK43_px, SJ1_AK43_py, SJ1_AK43_pz, SJ1_AK43_E;
         double SJ1_AK44_px, SJ1_AK44_py, SJ1_AK44_pz, SJ1_AK44_E;

         double SJ2_AK41_px, SJ2_AK41_py, SJ2_AK41_pz, SJ2_AK41_E;
         double SJ2_AK42_px, SJ2_AK42_py, SJ2_AK42_pz, SJ2_AK42_E;
         double SJ2_AK43_px, SJ2_AK43_py, SJ2_AK43_pz, SJ2_AK43_E;
         double SJ2_AK44_px, SJ2_AK44_py, SJ2_AK44_pz, SJ2_AK44_E;
      */

      
      for (auto iPJ=jetsFJ_jet.begin(); iPJ<jetsFJ_jet.end(); iPJ++)                             
      {


         if(iPJ->E()>50.)
         {
            SJ_50_px+=iPJ->px();SJ_50_py+=iPJ->py();SJ_50_pz+=iPJ->pz();SJ_50_E+=iPJ->E();
            SJ_nAK4_50[nSuperJets]++;
            tot_nAK4_50++;
         }
         if(iPJ->E()> 70.)
         {
            
            SJ_70_px+=iPJ->px();SJ_70_py+=iPJ->py();SJ_70_pz+=iPJ->pz();SJ_70_E+=iPJ->E();
            SJ_nAK4_70[nSuperJets]++;
            tot_nAK4_70++;
         }
         if(iPJ->E()>100)
         {
            SJ_100_px+=iPJ->px();SJ_100_py+=iPJ->py();SJ_100_pz+=iPJ->pz();SJ_100_E+=iPJ->E();
            diSuperJet_E_100 +=iPJ->E();diSuperJet_px_100+=iPJ->px();diSuperJet_py_100+=iPJ->py();diSuperJet_pz_100+=iPJ->pz();
            SJ_nAK4_100[nSuperJets]++; 
            
         }
         if(iPJ->E()>125)
         {
            SJ_125_px+=iPJ->px();SJ_125_py+=iPJ->py();SJ_125_pz+=iPJ->pz();SJ_125_E+=iPJ->E();
            SJ_nAK4_125[nSuperJets]++; 
           
         }
         if(iPJ->E()>150)
         {
            SJ_150_px+=iPJ->px();SJ_150_py+=iPJ->py();SJ_150_pz+=iPJ->pz();SJ_150_E+=iPJ->E();
            SJ_nAK4_150[nSuperJets]++; 
            
         }
         if(iPJ->E()>200)
         {
            SJ_200_px+=iPJ->px();SJ_200_py+=iPJ->py();SJ_200_pz+=iPJ->pz();SJ_200_E+=iPJ->E();
            SJ_nAK4_200[nSuperJets]++; 
           
         }
         if(iPJ->E()>300)
         {
            SJ_300_px+=iPJ->px();SJ_300_py+=iPJ->py();SJ_300_pz+=iPJ->pz();SJ_300_E+=iPJ->E();
            SJ_nAK4_300[nSuperJets]++; 
           

            // loop over selectedAK8_TLV
            // see if SJ designation matches current SJ
            // boost that jet to the SJ COM frame
            // check to see if this reclustered CA4 jet is within R = 0.8
            int selected_AK8_num = 0;
            for(auto iAK8_selected = selectedAK8_TLV.begin(); iAK8_selected!=selectedAK8_TLV.end();iAK8_selected++)
            {
               if(AK8_SJ_assignment[selected_AK8_num] == nSuperJets)
               {
                  TLorentzVector boosted_AK8(iAK8_selected->Px(),iAK8_selected->Py(),iAK8_selected->Pz(),iAK8_selected->E());
                  boosted_AK8.Boost(-superJetpx/superJetE,-superJetpy/superJetE,-superJetpz/superJetE); // boost the AK8 jet to its respective SJ COM frame
               
               }
               selected_AK8_num++;
            }

           
         }
         if(iPJ->E()>400)
         {
            SJ_400_px+=iPJ->px();SJ_400_py+=iPJ->py();SJ_400_pz+=iPJ->pz();SJ_400_E+=iPJ->E();
            SJ_nAK4_400[nSuperJets]++; 
            
         }
         if(iPJ->E()>600)
         {
            SJ_600_px+=iPJ->px();SJ_600_py+=iPJ->py();SJ_600_pz+=iPJ->pz();SJ_600_E+=iPJ->E();
            SJ_nAK4_600[nSuperJets]++; 
         
         }
         if(iPJ->E()>800)
         {
            SJ_800_px+=iPJ->px();SJ_800_py+=iPJ->py();SJ_800_pz+=iPJ->pz();SJ_800_E+=iPJ->E();
            SJ_nAK4_800[nSuperJets]++; 
           
         }
         if(iPJ->E()>1000)
         {
            SJ_1000_px+=iPJ->px();SJ_1000_py+=iPJ->py();SJ_1000_pz+=iPJ->pz();SJ_1000_E+=iPJ->E();
            SJ_nAK4_1000[nSuperJets]++; 
            
         }
         tot_mpp_AK4++;
      }

      SJ_mass_50[nSuperJets]= sqrt(pow(SJ_50_E,2)-pow(SJ_50_px,2)-pow(SJ_50_py,2)-pow(SJ_50_pz,2));
      SJ_mass_70[nSuperJets]= sqrt(pow(SJ_70_E,2)-pow(SJ_70_px,2)-pow(SJ_70_py,2)-pow(SJ_70_pz,2));
      SJ_mass_100[nSuperJets]= sqrt(pow(SJ_100_E,2)-pow(SJ_100_px,2)-pow(SJ_100_py,2)-pow(SJ_100_pz,2));
      SJ_mass_125[nSuperJets]= sqrt(pow(SJ_125_E,2)-pow(SJ_125_px,2)-pow(SJ_125_py,2)-pow(SJ_125_pz,2));
      SJ_mass_150[nSuperJets]= sqrt(pow(SJ_150_E,2)-pow(SJ_150_px,2)-pow(SJ_150_py,2)-pow(SJ_150_pz,2));
      SJ_mass_200[nSuperJets]= sqrt(pow(SJ_200_E,2)-pow(SJ_200_px,2)-pow(SJ_200_py,2)-pow(SJ_200_pz,2));
      SJ_mass_300[nSuperJets]= sqrt(pow(SJ_300_E,2)-pow(SJ_300_px,2)-pow(SJ_300_py,2)-pow(SJ_300_pz,2));
      SJ_mass_400[nSuperJets]= sqrt(pow(SJ_400_E,2)-pow(SJ_400_px,2)-pow(SJ_400_py,2)-pow(SJ_400_pz,2));
      SJ_mass_600[nSuperJets]= sqrt(pow(SJ_600_E,2)-pow(SJ_600_px,2)-pow(SJ_600_py,2)-pow(SJ_600_pz,2));
      SJ_mass_800[nSuperJets]= sqrt(pow(SJ_800_E,2)-pow(SJ_800_px,2)-pow(SJ_800_py,2)-pow(SJ_800_pz,2));
      SJ_mass_1000[nSuperJets]= sqrt(pow(SJ_1000_E,2)-pow(SJ_1000_px,2)-pow(SJ_1000_py,2)-pow(SJ_1000_pz,2));



      boostedSuperJetPart.clear();   //shouldn't be needed, just in case
      if(_verbose)std::cout << "ending superjet" << nSuperJets <<std::endl;
      nSuperJets++; 
   }
   if(_verbose)std::cout << "calculating disuperjet masses" <<std::endl;
   diSuperJet_mass = sqrt(pow(diSuperJet_E,2)-pow(diSuperJet_px,2)-pow(diSuperJet_py,2)-pow(diSuperJet_pz,2));
   diSuperJet_mass_100 = sqrt(pow(diSuperJet_E_100,2)-pow(diSuperJet_px_100,2)-pow(diSuperJet_py_100,2)-pow(diSuperJet_pz_100,2));
   if(_verbose)std::cout << "clear superjets"  <<std::endl;

   superJetOne.clear();
   superJetTwo.clear();

   // if ((nfatjets < 2) || ((nfatjet_pre < 1  )  ) )return; // RETURN cut     ///   ((dijetMassOne < 800.) || (dijetMassTwo < 800.) // normal selection
   if ((nAK8_ET100 < 2) || (nHeavyAK8_pt200_M10 < 1)) return; // RETURN cut

   nevents_200++;
  





   tree->Fill();

}   
DEFINE_FWK_MODULE(selectionStudier_200);
//_bottom_