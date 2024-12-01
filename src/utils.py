import matplotlib.pyplot as plt
import requests
import numpy as np
import re

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

def download_pcam():
    #gdown --folder https://drive.google.com/drive/folders/1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB
    pass

def get_info(dataloaders):
    for key in dataloaders.keys():
        print(key, ' Loader ',len(dataloaders[key]),' Set ', len(dataloaders[key].dataset))



def scrape():
    """Get the table with locations from the website"""
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd

    dataframes = []
    for i in range(20):
        url = f"http://gigadb.org/dataset/view/id/100439/Samples_page/{i}"
        print(f"Scraping {url}...")  
        response = requests.get(url)
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table') 
        if table:
            headers = [header.text.strip() for header in table.find_all('th')]
            rows = []
            for row in table.find_all('tr')[1:]:
                cells = [cell.text.strip() for cell in row.find_all('td')]
                rows.append(cells)
            df = pd.DataFrame(rows, columns=headers)
            df = df.iloc[:, [0, 3]] 
            dataframes.append(df)
        else:
            print(f"No table found on {url}.")
        final_df = pd.concat(dataframes, ignore_index=True)
        final_df.to_csv("combined_table.csv", index=False)
        print(final_df)


def get_scrape():
    """Get the wanted data from the scraped table"""
    import pandas as pd
    import re

    root = 'data/pcamv1/'
    csv_file = root+"combined_table.csv" 
    df = pd.read_csv(csv_file)

    extracted_data = []
    for index, row in df.iterrows():
        sample_id = row["Sample ID"]
        sample_attributes = row["Sample Attributes"]
        if sample_id.startswith("test_"):
            match = re.search(r"Sample storage location:(.+?)Sample", sample_attributes)
            if match:
                storage_location = match.group(1).strip()
                extracted_data.append({"Sample ID": sample_id, "Sample storage location": storage_location})
    extracted_df = pd.DataFrame(extracted_data)
    output_file = root+"extracted_data.csv"
    extracted_df.to_csv(output_file, index=False)
    print(f"Extracted data saved to {output_file}")


def denormalize_image(image, mean, std):
    """Apply on normalized images shape ex 0.876"""
    mean = np.array(mean)
    std = np.array(std)
    image = image * std + mean  
    return np.clip(image, 0, 1) # should be between 0,1

def show_image(image, path, plot = True):
    """Show image directly from dataloader"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] # from imagenet standard
    image = image.permute(1, 2, 0).cpu().numpy()
    image = denormalize_image(image, mean, std)
    image = (image * 255).astype(np.uint8)
    if plot:
        plt.imshow(image)
        plt.savefig(path)
    return image

def plot_waterbirds_result(our_accuracy, our_agreement, path):
    """Values should have form  agreement_data = {
        "KD": [68.7, 75.2, 72.8, 77.9],
        "e$^2$KD": [70.3, 76.4, 74.5, 79.3]
    }"""
    categories = ["700 epochs OOD", "700 epochs ID", "+5x training OOD", "+5x training ID"]
    methods = ["KD", "e$^2$KD"]
    # the paper
    reproduce_accuracy = {
        "KD": [31.1, 95.5, 37.9, 97.3],
        "e$^2$KD": [36.8, 97.4, 47.7, 98.8]
    }
    reproduce_agreement = {
        "KD": [61.8, 95.5, 69.2, 97.5],
        "e$^2$KD": [69.1, 97.6, 80.2, 98.8]
    }

    x = np.arange(len(categories))
    width = 0.35 
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'hspace': 0.3})

    def annotate_bars(ax, x_positions, our_values, paper_values):
        for i, (our, paper) in enumerate(zip(our_values, paper_values)):
            diff = our - paper
            highest_value = max(our, paper)
            ax.text(
                x_positions[i], highest_value + 0.5, 
                f"{our:.1f} ({'+' if diff > 0 else ''}{diff:.1f})",
                ha="center", fontsize=9, color="black"
            )

    axes[0].bar(x - width/2, our_accuracy["KD"], width, label="KD (Our)", color="lightgray", edgecolor="black")
    axes[0].bar(x + width/2, our_accuracy["e$^2$KD"], width, label="e$^2$KD (Our)", color="skyblue", edgecolor="black")

    axes[0].bar(x - width/2, reproduce_accuracy["KD"], width, label="KD (Paper)", color="none", edgecolor="black", hatch="...", alpha=0.5)
    axes[0].bar(x + width/2, reproduce_accuracy["e$^2$KD"], width, label="e$^2$KD (Paper)", color="none", edgecolor="black", hatch="...", alpha=0.5)

    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend(loc="lower right")

    annotate_bars(axes[0], x - width/2, our_accuracy["KD"], reproduce_accuracy["KD"])
    annotate_bars(axes[0], x + width/2, our_accuracy["e$^2$KD"], reproduce_accuracy["e$^2$KD"])

    axes[1].bar(x - width/2, our_agreement["KD"], width, label="KD (Our)", color="lightgray", edgecolor="black")
    axes[1].bar(x + width/2, our_agreement["e$^2$KD"], width, label="e$^2$KD (Our)", color="skyblue", edgecolor="black")

    axes[1].bar(x - width/2, reproduce_agreement["KD"], width, label="KD (Paper)", color="none", edgecolor="black", hatch="...", alpha=0.5)
    axes[1].bar(x + width/2, reproduce_agreement["e$^2$KD"], width, label="e$^2$KD (Paper)", color="none", edgecolor="black", hatch="...", alpha=0.5)

    axes[1].set_ylabel("Agreement")
    axes[1].set_title("Agreement")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)

    annotate_bars(axes[1], x - width/2, our_agreement["KD"], reproduce_agreement["KD"])
    annotate_bars(axes[1], x + width/2, our_agreement["e$^2$KD"], reproduce_agreement["e$^2$KD"])

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([0, 110])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('test.png')
    plt.savefig(path)
    plt.show()


def path2num(path):
    """ex waterbirds/student_model_waterbirds_kd_epoch_40.pth returns 40"""
    number = re.search(r"epoch_(\d+)", path)
    return int(number.group(1))


