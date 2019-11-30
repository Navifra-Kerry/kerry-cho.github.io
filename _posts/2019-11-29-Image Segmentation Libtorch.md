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

본 포스트를 읽기 전 사전에 고전적인 Image Segmentation과 Image Processing에 대한 기초 지식이 필요 합니다. 
해당 내용을 제가 설명 해드리는 것 보다, 전문적인 강의를 추천 드립니다. 

Microsoft에서는 무료 강의와 도큐먼트를 잘만들기로 유명 한데요 (저만 그렇게 생각하나요?) 정말 이렇게 해도 되나 싶은데 
PyTorch 포스팅을 하면서 Microsoft에서 만든 무료 교육을 추천 드립니다. 1달간 무료로 시청 하실 수 있는 것으로 알고 있고,
해당 사이트의 경우 인증서 같은 것을 발급해 주는데 그것을 받기 위해서 일정의 돈을 지급해야 하는 것으로 알고 있습니다. 

지금 추천드리는 강의보다 좋은 것들이 많이 있으니 공부 하실때 참고 하면 좋을것 같습니다.. 사이트 주소는 [여기](https://courses.edx.org/courses/course-v1:Microsoft+DEV290x+3T2019/course/) 입니다. 아 그리고 사이트의 강의 영어로 진행 됩니다. 영어 울렁증이 있으신 분들에게는 죄송합니다.....

그럼 시작하겠습니다.

# Introdution  
딥러닝을 이용한 Python 예제는 [여기](https://github.com/mrgloom/awesome-semantic-segmentation)에서 찾아 보실 수 있습니다. 
해당 사이트에 많은 예제들이 있지만 저는 PyTorch를 사용 했기 때문에 TorchVision Reference를 참고 하여 LibTorch 만들어 보았습니다.

그리고 데이터 셋의 소스는 [이친구](https://github.com/lsrock1/maskrcnn_benchmark.cpp)의 소스를 참고해서 제가 사용하기 편한 방법으로 변경 하였습니다.
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

데이터 셋은 MS COCO 데이터 셋을 사용 하였습니다.
데이터 셋은 클래스는 메인 객체와 2개의 하위 객체를 포함 하고 있습니다. 시각화를 위해 UML을 그려 보았습니다.
COCODataSet 클래스에서 멤버 클래스로 CocoDetection 클래스를 가지고  CocoDetection에서 CocoData셋의 JSON Parser인 CocoNote를 포함 하고 있습니다.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/Segmentation/DataSet.png" alt="DataSet">

COCODataSet 구조  
```c++
/*
COCODataSet은 객체 생성시 4개의 입력을 받습니다.
annFile  = annotation File의 전체 경로
root	 = 이미지 파일이 있는 경로 ,annotation에는 이미지 파일 명만 표시 되어 있습니다.
cat_list = 학습 시키길 원하는 카테고리 정보 0은 Background 이며 , 숫자로 입력됩니다
		   해당 숫자는 MS COCO 2017 카테고리 정보를 참고 하세요.
remove_images_without_annotations = 주석이 없는 이미지의 삭제 여부

Example
auto val_dataset = COCODataSet(data_dir + "annotations\\instances_val2017.json", data_dir + "val2017", true, { 0,17,18 })
	.map(torch::data::transforms::Stack<>());
const size_t va_dataset_size = val_dataset.size().value();

*/


class COCODataSet : public torch::data::Dataset<COCODataSet>
{
private:

	std::vector<torch::Tensor> states, labels;
	size_t ds_size;
	torch::data::transforms::Normalize<> normalizeChannels;
public:
	COCODataSet(std::string annFile, std::string root, bool remove_images_without_annotations 
		, std::vector<int> cat_list = std::vector<int>{});

	torch::data::Example<> get(size_t index) override;
	torch::optional<size_t> size() const override;

	rcnn::data::COCODetection _coco_detection;
	std::vector<int> _cat_list;
	std::map<int, int> _cat_idx;
};
// 하위 객체 설명 후에 더 자세히 설명 하겠습니다.
```

COCODetection
```c++

/*
COCODetection 의 경우 생성시 2개의 입력을 받습니다.
annFile  = annotation File의 전체 경로
root	 = 이미지 파일이 있는 경로 ,annotation에는 이미지 파일 명만 표시 되어 있습니다.
*/
namespace rcnn {
namespace data 
{

class COCODetection : public torch::data::datasets::Dataset<COCODetection, torch::data::Example<cv::Mat, std::vector<Annotation>>> 
{

public:
	COCODetection(std::string root, std::string annFile/*TODO transform=*/);
	torch::data::Example<cv::Mat, std::vector<Annotation>> get(size_t index) override;
	torch::optional<size_t> size() const override;

	std::string _root;
	COCONote _coco;
	std::vector<int> _ids;

	friend std::ostream& operator << (std::ostream& os, const COCODetection& bml);
};

}//data
}//rcnn

/*
이미지파일의 인덱스 정보를 가져와 해당 이미지의 Annotation 정보와 이미지파일을 로딩 하여 반환 합니다.
*/
torch::data::Example<cv::Mat, std::vector<Annotation>> COCODetection::get(size_t index) 
{
	int img_id = _ids.at(index);
	std::vector<int64_t> ann_ids = _coco.GetAnnIds(std::vector<int>{img_id}); //Image ID
	std::vector<Annotation> target = _coco.LoadAnns(ann_ids); // Load Anotations
	std::string path(_coco.LoadImgs(std::vector<int>{img_id})[0]._file_name); //LoadImage
	cv::Mat img = cv::imread(_root + "/" + path, cv::IMREAD_COLOR);

	if (img.rows == 0)
	{
		std::cout << "The image does not exist." << std::endl;
		std::cout << _root + "/" + path << std::endl;
		quick_exit(1);
	}


	torch::data::Example<cv::Mat, std::vector<Annotation>> value{ img, target };
	return value;
}

/*
전체 데이터 사이즈를 반환
*/
torch::optional<size_t> COCODetection::size() const
{
	return _ids.size();
}
```

COCONote
```c++

/*
Poco::JSON의 JSON을 사용해 annotation 정보를 읽어 드리는 클래스 입니다.
Poco는 해당 프로젝트를 위해 직접 빌드 하여 사용 하였습니다.
기회가 된다면 자세히 리뷰를 하는 포스팅을 진행 하겠습니다.
*/
struct COCONote
{
	COCONote(std::string annotation_file);
	COCONote();

	void Parse();
	void Parse(std::string annotation_file);

	std::vector<int64_t> GetAnnIds(const std::vector<int> imgIds = std::vector<int>{}, const std::vector<int> catIds = std::vector<int>{}, const std::vector<float> areaRng = 
...
};

/*
직접 Parsing을 진행 하는 함수 입니다.
*/
void COCONote::Parse()
{
#ifdef _DEBUG
	std::cout << "Parse...\n";
#endif
	if(_cocodataset->has("annotations"))//annotations 정보가 있는지 확인 합니다.
	{
		assert(_cocodataset->get("annotations").isArray()); //annotations이 Array객체가 아닐 경우 예외를 발생 시킵니다.

		Array::Ptr a = _cocodataset->get("annotations").extract<Array::Ptr>(); //annotations 을 ArryPtr 타입으로 변환 합니다.

		for(int i = 0; i < a->size(); i++)//Array Size 만큼 for를 진행 합니다.
		{
			Object::Ptr j = a->get(i).extract<Object::Ptr>(); //각 인덱스 별로 Object::Ptr변환 합니다.

			if(_imgToAnns.count(j->get("image_id").convert<int>()))//이미지의 id를 가져와 _imgToAnns에 있는지 비교합니다.
			{ // if it exists
				_imgToAnns[j->get("image_id").convert<int>()].emplace_back(j);// 존재 할경우에 해당 Key값에 집어 넣습니다.
			}
			else
			{
				_imgToAnns[j->get("image_id").convert<int>()] = std::vector<Annotation> {Annotation(j)};// 없을 경우에는 새로운 Annotation만들어 _imgToAnns에 담습니다.
			}
			
			_anns[static_cast<int64_t>(j->get("id").convert<int64_t>())] = Annotation(j);
		}
	}

	....
}
```

* Data Arguments  
이미지와 Annotation 정보는 COCODataSet Get 함수가 호출 될시에 진행 됩니다.

```c++

/*
함수가 상당히 길지만 상세하게 설명 하고 주의 깊게 보는게 좋을 것 같습니다.
*/
torch::data::Example<> COCODataSet::get(size_t idx)
{
	auto coco_data = _coco_detection.get(idx);
	cv::Mat img = coco_data.data;
	
	//현재 idx Annotation 정보를 가져와 내가 학습 하고자 하는 카테고리를 제외한 정보는 삭제 합니다.

	std::vector<Annotation> anno = coco_data.target;
	for (auto ann = anno.begin(); ann != anno.end();) 
	{
		if (std::find(_cat_list.begin(), _cat_list.end(), ann->_category_id) == _cat_list.end())
		{
			anno.erase(ann);
		}
		else
		{
			ann++;
		}
	}

	//Annotation 정보는 Polygon 형태의 자료형으로 되어 있습니다. 해당 정보를 
	// H * W 형태의 Matrix 타입의 구조로 타입 변환을 진행 하는 함수 입니다.
	// Matrix 정보에서 해당 카테고리 영역은 값이 1이고 , 나머지는 0의 값이 채워 집니다.
	// COCO DataSet의 Polygon 정보는  x1,y1,x2,y2,x3,y3,xn,yn의 형태로 double Array 타입으로 되어 있습니다.
	std::vector<int> cats;
	std::vector<std::vector<std::vector<double>>> polys;
	for (auto& obj : anno)
	{
		polys.push_back(obj._segmentation);//Annotation _segmentation정보만 가져 옵니다. PolyLines
		cats.push_back(_cat_idx[obj._category_id]);//위 Polygon의 Category ID를 가져 옵니다.
	}

	std::vector<torch::Tensor>  mask_tensors;

	//이미지와 마스크의 입력 사이즈는 480으로 변경 하기위한 Base Size 입니다.
	int base_size = 480;

	//Polygon To Mask Tensors
	for (int k= 0; k< polys.size(); k++)
	{
		//Polygon의 사이즈가 0 일 경우 리턴 합니다.
		if (polys[k].size() == 0) continue;
		
		//현재 로딩 된 이미지와 Base사이지를 비교해 scale 값을 구한 후 Polygon을 Resize 해줍니다.		
		transforms::polygon::Resize((double)base_size / (double)img.cols, (double)base_size / (double)img.rows, polys[k]);


		//coco API를 사용 하기위해서 Polygon 정보를 coco API 에서 사용하는 자료 형으로 변환 후 
		//Mask 정보를 리턴 받습니다.
		auto frPoly = coco::frPoly(polys[k], base_size, base_size);

		coco::RLEs Rs(1);

		coco::rleFrString(Rs._R, (char*)frPoly[0].counts.c_str(), std::get<0>(frPoly[0].size), std::get<1>(frPoly[0].size));
		coco::siz h = Rs._R[0].h, w = Rs._R[0].w, n = Rs._n;
		coco::Masks masks = coco::Masks(base_size, base_size, 1);

		coco::rleDecode(Rs._R, masks._mask, n);

		//coco API 변환한 Mask Size 만큼 비어 있는 Tensor 생성 합니다.
		int shape = h * w * n;
		torch::Tensor mask_tensor = torch::empty({ shape });

		float* data1 = mask_tensor.data_ptr<float>(); //해당 Tensor를 Dataptr 변경 후에 
		for (size_t i = 0; i < shape; ++i) {
			data1[i] = static_cast<float>(masks._mask[i] * cats[k]);//Category ID를 곱한 후 값을 복사 해 줍니다.
		}

		//Mask는 해당 Category 영역은 1, 아닐 경우 0 이기 때문에 Category ID 곱하면 고양이 일 경우 17로 변경 되고
		//아닌 영역은 0으로 채워 집니다.

		//Mask_tensor를 Category에 Mapping 후에 이미지 사이와 동일한 Matrix 형태로 변경 합니다
		// h * w 
		mask_tensor = mask_tensor.reshape({ static_cast<int64_t>(n),static_cast<int64_t>(w),
			static_cast<int64_t>(h) }).permute({ 2, 1, 0 }).squeeze(2);//fortran order h, w, n

		mask_tensors.push_back(mask_tensor);
	}
	
	// mask_tensors는 Vector를 Tensor Type로 변경 합니다.
	// n * h * w 형태로 변환 됩니다.
	auto mask_tensor = torch::stack(mask_tensors); 

	//n * h * w 형태의 Tensor를 하나로 합치게 됩니다.
	//예 4 * h * w -> h * w;
	// 동일한 영역의 값을 취하는 게 아니라 Max 값만 가져 옵니다.
	// 즉 서로 다른 카테고리 마스크를 하나로 합칩니다.
	torch::Tensor target, _;
	std::tie(target, _) = torch::max(mask_tensor, 0);
	
	//현재 로딩 된 이미지를 Resizng 합니다.
	cv::resize(img, img, cv::Size(base_size, base_size));


	//Random 값이 2의 배수일 경우 이미지와 target을 Horizental Flip을 진행 합니다.
	if (die(mersenne) % 2 == 0)
	{
		target = target.flip({ 1 });
		cv::flip(img, img, 1);
	}

	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);
	img_tensor = img_tensor.permute({ 2, 0, 1 });

	//이미지를 normalize 합니다.
	img_tensor = normalizeChannels(img_tensor);

	// 아래 코드는 이미와 Tensor가 제대로 입력이 되는지 확인 하기 위한 Debug  코드 입니다.
#if 0 // Debug Data Inputs
	std::cout << img_tensor.sizes() << std::endl;
	std::cout << target.sizes() << std::endl;

	cv::Mat bin_mask = cv::Mat::eye(target.size(0), target.size(1), CV_8UC1);
	target = target.clamp(0, 255).to(torch::kU8);
	target = target.to(torch::kCPU);
	std::memcpy(bin_mask.data, target.data_ptr(), sizeof(torch::kU8) * target.numel());

	uchar* data_ptr = (uchar*)bin_mask.data;

	for (int y = 0; y < bin_mask.rows; y++)
	{
		for (int x = 0; x < bin_mask.cols; x++)
		{
			if (data_ptr[y * bin_mask.cols + x] == 0)
			{
				continue;
			}
			else
			{
				data_ptr[y * bin_mask.cols + x]  = 255;
			}
		}
	}

	cv::imshow("Image", bin_mask);
	cv::imshow("Image2", img);
	cv::waitKey(0);
#endif

	//Image와 Tensor를 Tuple 형태로 반환 합니다.
	return { img_tensor.clone(), target.clone() };
}
```

* Data Loader  & Opimizer  

Data Loader Opimizer의 상세한 정보는 [TransferLearning](https://kerry-cho.github.io/TransferLearning-Libtorch/) 포스트를 참고 하시면 됩니다.
다른 점만 설명 하겠습니다.  
  
Segmentation에서는 Backbone이 존재하고 해당 모델의 경우 Aux 라는 옵션이 있기 때문에 그에 따라 Optimizer에서 학습 해야 하는  
학습 파라메터가 조금 씩 다릅니다.

```c++

/*
TransferLearning 크게 다른 점은 Backbone 파라메터와 _classifier의 파라메터 
aux_classifier파라메터를 각각 std::vector<torch::Tensor> 담아 주어야 한다는 점입니다.
해당 예제는 aux_classifier를 사용 하지 않기 때문에 생략 하였습니다.
*/
std::vector<torch::Tensor> trainable_params;	
auto params = segnet->_classifier->named_parameters(true /*recurse*/);
for (auto& param : params)
{
	auto layer_name = param.key();

	if (param.value().requires_grad())
	{
		trainable_params.push_back(param.value());
	}
}

params = segnet->_backbone->named_parameters(true /*recurse*/);
for (auto& param : params)
{
	if (param.value().requires_grad())
	{
		trainable_params.push_back(param.value());
	}
}

torch::optim::SGD optimizer(trainable_params, torch::optim::SGDOptions(0.01 /*learning rate*/).momentum(0.9).weight_decay(1e-4));
```


* Training Loop & test Loop

학습과 테스트는 TransferLearning 크게 다르지 않습니다. 따라서 Loss를 구하는 함수에 대해서만 설명 드리고 나머지 부분은 
[TransferLearning](https://kerry-cho.github.io/TransferLearning-Libtorch/) 포스트를 참고 하시면 됩니다.


loss함수

```c++

/*
loss 함수는 Classcification처럼 CrossEntropy를 구하는데요. Segmentation의 경우 Pixel 전체를 비교 한다고 생각 하시면 됩니다.
해당 함수는 LibTorch에 존재 하지 않아서 Network 출력 값을 log_softmax 취한 후 nll_loss2d 함수로 계산 합니다.
*/
torch::Tensor criterion(
	std::unordered_map<std::string, torch::Tensor> inputs, torch::Tensor target)
{
	std::map<std::string, torch::Tensor> losses;

	for (auto loss : inputs)
	{
		losses[loss.first] = torch::nll_loss2d(torch::log_softmax(loss.second, 1), target, {}, 1, 255);
	}

	if (losses.size() == 1)
	{
		return losses["out"];
	}

	return losses["out"] + 0.5 * losses["aux"];
}
```

## Conclusion
지금까지 LibTorch를 이용한 Segmentation 대해서 이야기를 하였습니다. 여기 까지 읽어 주신것을 감사 드립니다.
Segmentation에 대해 이론적으로 접근 하기 보다 LibTorch를 활용 한 방법에 대해서 설명을 집중 하였습니다. 포스팅 초반에 Segmentation이라는 
사전지식이 필요 하다고 했는데 실제 해당 내용에 대한 설명은 진행 하지 않은 것 같네요. 하지만 Segmentation이 무엇을 하는 것이란 걸
알고 해당 포스트를 읽는 것이 중요 하다고 생각하였습니다.
궁금 한 내용이나, 잘못 된 점이 있다면 답글에 남겨 주시면 감사 하겠습니다. 
다음 이야기는 Microsoft에서 진행 하는 OpenSource 프로젝트중 VoTT에 대한 리뷰를 진행 하겠습니다(Visual Object Tagging Tool)
감사합니다.
