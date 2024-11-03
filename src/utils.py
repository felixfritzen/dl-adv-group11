import matplotlib.pyplot as plt
import requests


def key2key(dict1, dict2):
    """If targets are the same, maps key to key"""
    feature_to_key_dict2 = {}
    for key, value in dict2.items():
        feature_to_key_dict2[tuple(value)] = key
    mapping = {}
    for key1, value1 in dict1.items():
        feature_tuple = tuple(value1)
        if feature_tuple in feature_to_key_dict2:
            mapping[key1] = feature_to_key_dict2[feature_tuple]
    mapping = {str(key): int(value) for key, value in mapping.items()}
    return mapping


def show_image(image):
    #plt.imshow(imagenet.idx2image(1))
    plt.imshow(image.permute(1, 2, 0))
    plt.savefig("figs/plot.png") 

def accuracy(predictions, ground_truth):
    correct = (predictions == ground_truth).sum().item()
    total = len(predictions)
    accuracy = correct / total * 100
    return accuracy


def downloadVOC():
    # Login URL and credentials
    login_url = "http://host.robots.ox.ac.uk/accounts/login/"
    download_url = "http://host.robots.ox.ac.uk/eval/downloads/VOC2012test.tar"
    session = requests.Session()

    # Login payload
    payload = {
        "username": "felixfritzen",
        "password": "nozkot-jixbo3-patqyW"
    }

    # Post login data
    session.post(login_url, data=payload)

    # Download file
    with session.get(download_url, stream=True) as response:
        with open("VOC2012test.tar", "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)


def get_info(dataloaders):
    for key in dataloaders.keys():
        print(key, ' Loader ',len(dataloaders[key]),' Set ', len(dataloaders[key].dataset))

