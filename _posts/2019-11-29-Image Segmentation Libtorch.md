---
title: "Image Segmentation Libtorch"
use_math: true
tags: [PyTorch, Libtorch, Segmentation, DeepLearning]
header:
---

안녕하세요. 조대희 입니다.
블로그 방문을 환영 합니다.

이번에 소개 해드릴 내용은 PyTorch의 C++ Frontend 인 LibTorch를 활용한 Image Segmentation을 하는 법에 대한 내용 입니다.
모든 소스는 [여기](https://github.com/kerry-Cho/SemanticSegmentation-Libtorch) 저장소에 있습니다. 

본 포스트를 읽기 전에 사전에 고전적인 Image Segmentation과 Image Processing에 대한 기초 지식이 필요 합니다. 
해당 내용을 제가 설명 해드리는 것 보다, 전문적인 강의를 추천 드립니다. 

Microsoft에서는 무료 강의와 도큐먼트를 잘만들기로 유명 한데요 (저만 그렇게 생각하나요?) 정말 이렇게 해도 되나 싶은데 
PyTorch 포스팅을 하면서 Microsoft에서 만든 무료 교육을 추천 드립니다. 1달간 무료로 시청 하실 수 있는 것으로 알고 있고,
해당 사이트의 경우 인증서 같은 것을 발급해 주는데 그것을 받기 위해서 일정의 돈을 지급해야 하는 것으로 알고 있습니다. 

지금 추천드리는 강의보다 좋은 것들이 많이 있으니 공부 하실때 참고 하면 좋을것 같습니다.. 사이트 주소는 [여기](https://courses.edx.org/courses/course-v1:Microsoft+DEV290x+3T2019/course/) 입니다. 아 그리고 사이트의 강의 영어로 진행 됩니다. 영어 울렁증이 있으신 분들에게는 죄송합니다.....

그럼 시작하겠습니다.

# Introdution  
딥러닝을 이용한 Python 예제는 [여기](https://github.com/mrgloom/awesome-semantic-segmentation)에서 찾아 보실 수 있습니다. 
해당 사이트에 많은 예제들이 있지만 저는 PyTorch를 사용 했기 때문에 TorchVision Reference를 참고 하여 LibTorch 만들어 보았습니다.

그리고 데이터 셋의 소스 [이친구](https://github.com/lsrock1/maskrcnn_benchmark.cpp)의 소스를 참고해서 제가 사용하기 편한 방법으로 변경 하였습니다.
위의 친구가 지금 열심히 MaskRCNN LibTorch 버전을 만들고 있던데 다 만들어 지면 저것 또한 리뷰를 하는 시간을 가져 볼께요...아니면 제가 만들어서!!?

# Install Dependency  
의존성 파일들은 [TransferLearning](https://kerry-cho.github.io/TransferLearning-Libtorch/) 포스트를 참고 하시면 됩니다.

# Network Model  
* Backbone  
Backbone 같은 경우 TorchVision의 Resnet 모델 가져와서 조금 수정 하였습니다. Segmentation의 경우 Dilate Convolution을 진행 해야 하고 
모듈 전체를 가져와 하나의 네트웍으로 구성 해야 하기 때문에 수정을 진행 했습니다. 
물론 Python 모델과 호환이 되게끔 수정 했습니다. 

중요한 내용은 아래와 같습니다.
```c++
template <typename Block>
struct ResNetImpl : torch::nn::Module
 {
    ...
	int64_t groups, base_width, inplanes;
	int64_t _dilation;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm bn1;
	torch::nn::Functional max_pool1; 
    /*
        BackBone으로 사용 하기 위해서는 max_pool을 API로 호출 하면
        Segmntation 모델에서 사용 할 수 없기 때문에 Functional 멤버 변수로 선언 해줍니다.
    */
	torch::nn::Linear fc;
	torch::nn::Sequential layer1, layer2, layer3, layer4;
    ...
}

template <typename Block>
torch::Tensor ResNetImpl<Block>::forward(torch::Tensor x) {
	x = conv1->forward(x);
	x = bn1->forward(x).relu_();
	/*
    기존 코드 
    x = torch::max_pool2d(x, 3, 2, 1);
    */
    x = max_pool1->forward(x);

	x = layer1->forward(x);
	x = layer2->forward(x);
	x = layer3->forward(x);
	x = layer4->forward(x);

	x = torch::adaptive_avg_pool2d(x, { 1, 1 }); //이 아이는 사용 하지 않기 때문에 멤버 함수로 선언 하지 않았어요...
	x = x.view({ x.sizes()[0], -1 });
	x = fc->forward(x);

	return x;
}
```
* IntermediateLayerGetter  
Python PyTorch에서는 IntermediateLayer의 경우 모든 Network을 입력 받아 원하는 레이어의 출력만 List에 담아서 리턴을 해주는 클래스인데요.
저 같은 경우에 현재는 resnet만 입력을 받아서 출력을 받는 형태로 되어 있습니다. 
여기 저기 물어 봤지만 anyModule 사용해서 하면 된다고 하는데 그렇게 할 경우에 weight값을 복사하기가 불편해 예제의 코드 처럼 구현 했습니다.

```c++

/*
생성자에서 Resnet을 입력 받아서 모듈을 OrderedDict 담아 놓고 해당 모듈은 child 모델로 등록 합니다.
사용 방법은 아래와 같아요.

ResNet101 Resnet;	
torch::load(Resnet, "resnet101_Python.pt");

_backbone = IntermediateLayerGetter(IntermediateLayerGetterImpl(std::move(Resnet), {"layer3","layer4"}));
*/
class IntermediateLayerGetterImpl : public torch::nn::Module
{
public:

	template<typename Net>
	IntermediateLayerGetterImpl(Net  Module, std::vector<std::string> return_layers)
	{
		for (auto children : Module->named_children())
		{
			if (children.key() == "fc") continue;
		
			_module.insert(children.key(), std::move(children.value()));
			register_module(children.key(), _module[children.key()]);
		}

		_return_layers.swap(return_layers);
	}

	~IntermediateLayerGetterImpl();

	std::vector<torch::Tensor>  forward(torch::Tensor x);

private:
	torch::OrderedDict<std::string, std::shared_ptr<Module>> _module;
	std::vector<std::string> _return_layers;
};

/*
아래의 forward 때문에 모든 네트웍을 구현 할수가 없었어요. 시간이 없었습니다?
torch::nn::Module을 as 함수를 통해서 Typecasting 지원 하는데요. 넘겨 받은 모델의 Type을 알 수가 없어서 한땀 한땀 변환 해줬습니다.
좋은 방법이 있다면 댓글로 달아 주시면 감사 하겠습니다.
이렇게 사용해도 성능에는 이상이 없습니다.
보기가 아름 답지 않을 뿐이죠....
*/
std::vector<torch::Tensor> IntermediateLayerGetterImpl::forward(torch::Tensor x)
{
	std::vector<torch::Tensor> results;

	x = _module["conv1"]->as<torch::nn::Conv2d>()->forward(x);	
	x = _module["bn1"]->as<torch::nn::BatchNorm>()->forward(x).relu_();
	x = _module["max_pool1"]->as<torch::nn::Functional>()->forward(x);

	x = _module["layer1"]->as<torch::nn::Sequential>()->forward(x);
	x = _module["layer2"]->as<torch::nn::Sequential>()->forward(x);
	x = _module["layer3"]->as<torch::nn::Sequential>()->forward(x);
	results.push_back(x);
	x = _module["layer4"]->as<torch::nn::Sequential>()->forward(x);
	results.push_back(x);
	
	return results;
}

```

* SegmentationModel  
SegmentationModel은 fcn과 deeplabv3 버전만 구현을 하였습니다. 이 예제를 보고 난후에 여러분은 다른 모델도 만들 자신감이 생기 실 겁니다.
여러분이 직접 구현 할 수 있게 나머지는 남겨 놓았습니다. PullRequest를 보내 주시면 감사 합니다.
멤버 함수로 모델을 선택 할 수 있게 구현 하였습니다.  


```c++

/*
    사용 법은 아래와 같아요.
	SegmentationModel segnet;
	segnet->deeplabv3_resnet101(false, class_num);
	segnet->to(device);
*/

class SegmentationModelImpl :public torch::nn::Module
{
public:
	SegmentationModelImpl();
	~SegmentationModelImpl();

public:
	void fcn_resnet101(bool pretrained = false, int64_t num_classes = 21, bool aux = true);
	void fcn_resnet50(bool pretrained = false, int64_t num_classes = 21, bool aux = true);

	void deeplabv3_resnet101(bool pretrained = false, int64_t num_classes = 21, bool aux = true);
	void deeplabv3_resnet50(bool pretrained = false, int64_t num_classes = 21, bool aux = true);

	std::unordered_map<std::string, torch::Tensor> forward(torch::Tensor x);
	IntermediateLayerGetter _backbone{ nullptr };


	torch::nn::Sequential _classifier{ nullptr };
	torch::nn::Sequential _aux_classifier{ nullptr };
	torch::nn::Sequential _make_FCNHead(int64_t in_channels, int64_t channels);
	bool _aux;
};

/*
DeeplabV3 3기준으로 설명을 드릴께요.
*/
void SegmentationModelImpl::deeplabv3_resnet101(bool pretrained, int64_t num_classes, bool aux)
{
	int64_t in_channels = 2048; // resnet의 4번째 레이어의 출력 값이 2048 * 60 * 60 이기때문에 입력은 2048 입니다.

    /*
        _classifier 실제로 Segmentation을 진행 하는 모델 입니다. 
        해당 모델은 논문을 참고 하시는게 도움이 되실꺼에요. 어줍짢은 지식의 전달 보다 사용 법에 집중 할게요.
        코드는 아래와 같이 선언 합니다.
    */
	_classifier = torch::nn::Sequential
	(
		ASPP(ASPPImpl(2048, { 12,24,36 })), 
		torch::nn::Conv2d(
			torch::nn::Conv2dOptions(256, 256, 3).padding(1).with_bias(false)),
		torch::nn::BatchNorm(
			torch::nn::BatchNormOptions(256).eps(0.001).momentum(0.01)),
		torch::nn::Functional(torch::relu),
		torch::nn::Conv2d(
			torch::nn::Conv2dOptions(256, num_classes, 1))
	);

    /*
     aux란 resnet의 3번째 레이어의 출력 값을 Segmentation에 활용 할지 여부인데요. 저같은 경우에는 사용하지 않았습니다.
    */
    
	if (aux != false)
	{
		_aux = aux;
		_aux_classifier = _make_FCNHead(1024, num_classes);
	}

    // 아래는 위에서 설명 드렸네요.
	ResNet101 Resnet;	
	torch::load(Resnet, "resnet101_Python.pt");

	_backbone = IntermediateLayerGetter(IntermediateLayerGetterImpl(std::move(Resnet), {"layer3","layer4"}));

	register_module("backbone", _backbone);
	register_module("classifier", _classifier);
	register_module("aux_classifier", _aux_classifier);
}

/*
 SegmentationModelImpl::forward는 출력인 2개 일 수도 있고 1개 일 수도 있습니다.
 std::unordered_map<std::string, torch::Tensor> 형태로 출력 되고 Key 값을 out과 aux 리턴 합니다.
 
 이미지를 _backbone에 forward 후 출력 값을 _classifier forward 후 출력 된 값을 
 이미지 사이즈 만큼 다시 upsample 하여 출력 합니다. aux를 사용 할 경우에 _aux_classifier 출력 값도 리턴 합니다.

*/
std::unordered_map<std::string, torch::Tensor> SegmentationModelImpl::forward(torch::Tensor x)
{
	std::unordered_map<std::string, torch::Tensor> result;

	int64_t h = x.size(2), w = x.size(3);

	auto feature = _backbone->forward(x);

	x = feature[1];

	x = _classifier->forward(x);
	x = torch::upsample_bilinear2d(x, { h,w }, false);
	result.insert(std::make_pair("out", x));

	if (_aux == true)
	{
		x = feature[0];
		x = _aux_classifier->forward(x);
		x = torch::upsample_bilinear2d(x, { h,w }, false);
		result.insert(std::make_pair("aux", x));
	}

	return result;
}  

```

# Convert Python Model Parameter to C++  
Convert Python Model [TransferLearning](https://kerry-cho.github.io/TransferLearning-Libtorch/) 포스트를 참고 하시면 됩니다.

# DataSet  
