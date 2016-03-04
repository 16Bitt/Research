#include <GClasses/GActivation.h>
#include <GClasses/GHolders.h>
#include <GClasses/GMatrix.h>
#include <GClasses/GNeuralNet.h>
#include <GClasses/GRand.h>
#include <GClasses/GTransform.h>

using namespace GClasses;

#define NUM_COLUMNS_SPLIT	1
#define EPOCH_SIZE		1
#define NUM_EPOCHS		10000
#define NUM_NODES			50

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
	
	//Now we need to normalize our inputs and outputs
	for(int i = 0; i < pRealInputs->cols(); i++){
		double min = pRealInputs->columnMin(i);
		double max = pRealInputs->columnMax(i);

		if(min == max)
			continue;

		pRealInputs->normalizeColumn(i, min, max);
	}

	for(int i = 0; i < pRealOutputs->cols(); i++){
		double min = pRealOutputs->columnMin(i);
		double max = pRealOutputs->columnMax(i);
		
		if(min == max)
			continue;

		pRealOutputs->normalizeColumn(i, min, max);
	}
	
	//Split our data into testing and training portions
	GRand r(0);
	GDataRowSplitter rs(*pRealInputs, *pRealOutputs, r, 1000);
	const GMatrix& trainingFeatures	= rs.features1();
	const GMatrix& trainingLabels	= rs.labels1();
	const GMatrix& testingFeatures	= rs.features2();
	const GMatrix& testingLabels	= rs.labels2();

	//Make our network with 4 layers
	GNeuralNet nn;
	nn.addLayer(new GLayerClassic(40, NUM_NODES, new GActivationTanH()));
	nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, NUM_NODES, new GActivationTanH()));
	nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, NUM_NODES, new GActivationTanH()));
	nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, NUM_COLUMNS_SPLIT, new GActivationTanH()));

	//Change our learning rate
	nn.setLearningRate(0.05);
	
	//Start our learning
	nn.beginIncrementalLearning(trainingFeatures, trainingLabels);

	//Make a random iterator
	GRandomIndexIterator ii(trainingFeatures.rows(), nn.rand());

	std::cout << "@RELATION neural_net_accuracy\n";
	std::cout << "@ATTRIBUTE epoch\n";
	std::cout << "@ATTRIBUTE sse\n";
	std::cout << "@ATTRIBUTE mse\n";
	std::cout << "@ATTRIBUTE rmse\n";
	std::cout << "@DATA\n";

	for(int i = 0; i < NUM_EPOCHS; i++){
		double sse	= nn.sumSquaredError(testingFeatures, testingLabels);
		double mse	= sse / testingLabels.rows();
		double rmse	= sqrt(mse);
		
		if(i % EPOCH_SIZE == 0){
			std::cout << i << ", " << sse << ", " << mse << ", " << rmse << "\n";
			std::cout.flush();
		}

		ii.reset();
		size_t index;
		while(ii.next(index))
			nn.trainIncremental(trainingFeatures[index], trainingLabels[index]);
	}

	delete(pRealInputs);
	delete(pRealOutputs);
	return EXIT_SUCCESS;
}
