---
title: "TransferLearning Libtorch"
use_math: true
tags: [PyTorch, Libtorch, TransferLearning, DeepLearning]
header:
---

안녕하세요. 조대희 입니다.
블로그 방문을 환영 합니다.

첫 번째로 소개 해드릴 내용은 PyTorch의 C++ Frontend 인 LibTorch를 활용하여 TransferLearning을 하는 법에 대한 내용 입니다.
모든 소스는 여기 저장소에 있습니다. 

그럼 지금 부터 시작 합니다.

# Introdution
Libtorch는 Pytorch의 C++ Frontend로 Python API와 동일 구조의 인터페이스를 가집니다. 현재까지는 Python 과 동일 한 수준의 API를 
제공 하지 않지만 1.4 이후에는 더욱 더 많은 API 제공 할 것으로 보입니다. 
```c++
1.3 이하 버전에서는 아래의 코드 처럼 예외처리 시에 unknow Exception이라는 메세지를 많이 보게 됩니다...
try
{
...
}
catch(std::Exception ex)
{
....
}
```
이제 부터 Python과 동일 한 구조로 작성 된 모델과 학습 방법에 대해 자세히 소개 해보겠습니다.

# Install Dependency
모든 의존성 파일은 설치 스크립를 작성해 놓았습니다.
해당 스크립트 Azure와 Github Actions CI 에서 많이 활용 될 수 있는데 다음번에 자세한 포스트를 진행 하겠습니다.
스크립트는 CNTK의 설치 스크립트를 예제의 환경에 맞게 수정 하였고 죄송하지만 윈도우 유저가 아닐 경우 직접 설치 하셔 합니다.

저장소에서 샘플 코드를 다운 받으시고 아래의 경로 install.bat 파일을 실행 주시면 모든게 자동으로 설치 됩니다.
단 CUDA를 사용 하기 위해서는 CUDA를 직접 설치 하셔야 합니다.

또한 설치는 D:\Local 폴더 안에 설치 됩니다. D드라이브가 없을 경우 환경 설정을 별도로 해주셔야 합니다.

```
transfer-learning
 - Scripts
    -install.bat
```

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/transferlearning/Install.png" alt="image">
    
# Network Model
 Resnet 모델을 사용 하였습니다. 해당 모델은 Torchvision C++ 모델을 가져와 사용 하였습니다. 
 해당 모델을 사용 한 이유는 네트웍의 파라메터 값을 손쉽게 복사 하기 위함 입니다.
 
# Convert Python Model Parameter to C++
JIT 스크립트를 활용해 C++ 용 모델 파일을 저장 하여 C++로 생성 된 모델에 값을 로드 하는 방법을 사용 합니다.
네트웍 구조가 동일 하지 않을 경우 Unknow Exception이라는 에러 메세지를 보게 될 것 입니다.(Torchvision모델을 사용 합시다.)
예제 코드는 아래와 같습니다.

```python
#Python Model Convert JIT 스크립트
import torch
from torchvision import models

# Download and load the pre-trained model
model = models.resnet18(pretrained=True)

# Set upgrading the gradients to False
for param in model.parameters():
	param.requires_grad = False

example_input = torch.rand(1, 3, 224, 224)
script_module = torch.jit.trace(model, example_input)
script_module.save('resnet18_Python.pt')
```

```C++
//Load Parameter
ResNet18 network;
torch::load(network, "resnet18_Python.pt");
```

* FC Layer  
클래스 갯수를 학습 시킬 모델 만큼 변경 해주어야 합니다.
Python과 다르게 C++에서는 할당 및 등록을 별도로 해주어야 하고 새로 등록 하기 위해서는 등록 된 파라메테를 해제 후 제 등록 해야합니다.
코드는 아래와 같습니다.

Python
```Python
  model_ft = models.resnet18(pretrained=True)
  num_ftrs = model_ft.fc.in_features
  # Here the size of each output sample is set to 2.
  # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
  model_ft.fc = nn.Linear(num_ftrs, 2)
```

C++
```c++
  ResNet18 network;
	torch::load(network, "../../Model/resnet18_Python.pt");

  //해제
	network->unregister_module("fc");

  //SharedPtr 다시 생성
	network->fc = torch::nn::Linear(torch::nn::LinearImpl(512, 2));	

  //등록
	network->register_module("fc", network->fc);
```

# DataSet  
* 데이터 셋은 Torch 모델에서 제공 하는 데이터셋 클래스를 상속 받은 후 두가지 함수를 Overriding 해주셔야 합니다.


```c++
class ImageNetDataSet : public torch::data::Dataset<ImageNetDataSet /*,Example 자료형*/>
{
private:
	/* data */
	// Should be 2 tensors
	std::vector<torch::Tensor> states, labels;
	size_t ds_size;
public:
	ImageNetDataSet(std::string map_file)
	{
		std::tie(states, labels) = read_map(map_file);
		ds_size = states.size();
	};

// Get 함수의 리턴 값을 상속 된 클래스의 템플릿 구조에 따라 달라 질 수 있습니다.
// ImageNetDataSet의 경우 ImageNetDataSet 구조를 입력 하고 리턴 타입에 대해서 정의 하지 않았기 때문에
// 기본 리턴 값은 Example의 구조로 데이터가 리턴 됩니다.
	torch::data::Example<> get(size_t index) override {
		/* This should return {torch::Tensor, torch::Tensor} */
		torch::Tensor sample_img = states.at(index);
		torch::Tensor sample_label = labels.at(index);
		return { sample_img.clone(), sample_label.clone() };
	};

  //데이터 셋의 전체 사이즈를 리턴 해주어야 합니다.
	torch::optional<size_t> size() const override {
		return ds_size;
	};
};

//Example의 기본 형은 아래 처럼 튜플 형태의 자료 형으로
// Data Tensor와 Target Tensor를 리턴 합니다.
template <typename Data = Tensor, typename Target = Tensor>
struct Example {
  using DataType = Data;
  using TargetType = Target;

  Example() = default;
  Example(Data data, Target target)
      : data(std::move(data)), target(std::move(target)) {}

  Data data;
  Target target;
};
```
* image Arguments  
이미지 입력 전처리의 경우 OpenCV를 사용 하였습니다.

```c++
torch::Tensor read_data(std::string location) {
	/*
	 Function to return image read at location given as type torch::Tensor
	 Resizes image to (224, 224, 3)
	 Parameters
	 ===========
	 1. location (std::string type) - required to load image from the location

	 Returns
	 ===========
	 torch::Tensor type - image read as tensor
	*/
  /*Nomalize는 생략 되었고 BGR to RGB도 생략 되었습니다.(해당 코드는 Torch의 Example 코드를 사용 했습니다.)
  필요에 따라서 Nomalize와 이미지 Flip, Crop등의 처리도 여기서 진행 후 Tensor로 변환 하면 됩니다.
  */
	cv::Mat img = cv::imread(location, 1);
	cv::resize(img, img, cv::Size(224, 224), cv::INTER_CUBIC);
	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);
	img_tensor = img_tensor.permute({ 2, 0, 1 });
	return img_tensor.clone();
}
```

* 데이터 셋 입력 받기  
데이터 셋의 경우 MapFile 형태로 입력을 받아 한줄 한줄 Parsing하여 사용 하였습니다.

```c++
#Mapfile 

'이미지경로\t클래스인덱스'
..\..\Sample\test\cats\cat.4001.jpg	0
..\..\Sample\test\cats\cat.4002.jpg	0
..\..\Sample\test\cats\cat.4003.jpg	0
..\..\Sample\test\cats\cat.4004.jpg	0
..\..\Sample\test\cats\cat.4005.jpg	0
..\..\Sample\test\cats\cat.4006.jpg	0
..\..\Sample\test\cats\cat.4007.jpg	0
..\..\Sample\test\cats\cat.4008.jpg	0
..\..\Sample\test\cats\cat.4009.jpg	0
..\..\Sample\test\cats\cat.4010.jpg	0

#code
	std::ifstream stream;
	stream.open(map_file);

	message_assert(stream.is_open(), "error: failed to read info file \""
		<< map_file << "\"");

	std::string path, id;
	std::vector<torch::Tensor> states , labels;

	while (true)
	{
		path.clear();
		stream >> path >> id;
		if (path.empty()) break;

		torch::Tensor img = read_data(path);
		states.push_back(img);

		torch::Tensor label = read_label(std::stoi(id));
		labels.push_back(label);


		if (stream.peek() == EOF || stream.eof()) break;
	}

	stream.close();
```

* Label to Tensor  
라벨의 경우 클래스 인덱스로 입력 해주시면 됩니다  

```c++
torch::Tensor read_label(int label) {
	/*
	 Function to return label from int (0, 1 for binary and 0, 1, ..., n-1 for n-class classification) as type torch::Tensor
	 Parameters
	 ===========
	 1. label (int type) - required to convert int to tensor

	 Returns
	 ===========
	 torch::Tensor type - label read as tensor
	*/
	torch::Tensor label_tensor = torch::full({ 1 }, label);
	return label_tensor.clone();
}
```

* Data Loader


## Conclusion
