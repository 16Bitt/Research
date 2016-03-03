#include <GClasses/GActivation.h>
#include <GClasses/GHolders.h>
#include <GClasses/GMatrix.h>
#include <GClasses/GNeuralNet.h>
#include <GClasses/GRand.h>
#include <GClasses/GTransform.h>

using namespace GClasses;

#define USE_AUTOFILTER
#define NUM_COLUMNS_SPLIT	1

int main(int argc, char** argv){
	//Load the dataset into memory
	GMatrix data;
	data.loadArff("auth.arff");

	//Split the dataset
	GDataColSplitter splitter(data, NUM_COLUMNS_SPLIT);
	const GMatrix& inputs	= splitter.features();
	const GMatrix& outputs	= splitter.labels();

	//Transform the TRUE/FALSE columns to continuous values
	GNominalToCat nc_outputs;
	nc_outputs.train(outputs);
	GMatrix* pRealOutputs = nc_outputs.transformBatch(outputs);
	GNominalToCat nc_inputs;
	nc_inputs.train(inputs);
	GMatrix* pRealInputs = nc_inputs.transformBatch(inputs);

	//Split our data into testing and training portions
	GRand r(0);
	GDataRowSplitter rs(*pRealInputs, *pRealOutputs, r, 3000);
	const GMatrix& trainingFeatures	= rs.features1();
	const GMatrix& trainingLabels	= rs.labels1();
	const GMatrix& testingFeatures	= rs.features2();
	const GMatrix& testingLabels	= rs.labels2();


	//Make our network with 4 layers
	GNeuralNet nn;
	nn.addLayer(new GLayerClassic(20, 20, new GActivationTanH()));
	nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, 20, new GActivationTanH()));
	nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, 20, new GActivationTanH()));
	nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, 20, new GActivationTanH()));
	nn.setImprovementThresh(0.001);
	nn.setWindowSize(1000);
	
	#ifdef USE_AUTOFILTER
	//Use an autofilter to train our data
	GAutoFilter af(&nn, false);
	af.train(inputs, outputs);
	double missed = af.sumSquaredError(inputs, outputs);
	std::cout << "Misclassified " << missed << " out of " << outputs.rows() << "\n";
	#else
	nn.setLearningRate(0.05);
	nn.train(trainingFeatures, trainingLabels);

	double sse	= nn.sumSquaredError(testingFeatures, testingLabels);
	double mse	= sse / testingLabels.rows();
	double rmse	= sqrt(mse);
	std::cout << "RMSE = " << rmse << "\n";
	#endif
	
	delete(pRealInputs);
	delete(pRealOutputs);
	return EXIT_SUCCESS;
}
