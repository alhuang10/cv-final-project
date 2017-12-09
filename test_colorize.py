# Generate colorized output serialized Torch model weights

from net import *
import sys

WEIGHT_PATH = sys.argv[1]
NUM_IMAGES = 20
TEMPERATURE = 0.38

sabrina = SabrinaNet()

use_cuda = torch.cuda.is_available()

if use_cuda:
    print("Using CUDA")
    sabrina.cuda()

sabrina.load_state_dict(torch.load(WEIGHT_PATH))

train_data_transform = Compose([Resize((256,256)), PILToTensor()])

bin2ab = {v:k for (k,v) in ab2bin.items()}

with open('bin2ab.p','wb') as f:
    pickle.dump(bin2ab, f)

print("Creating DataLoader")

train_images = ImageNet("ILSVRC2012_img_val/", transform=train_data_transform)
trainloader = torch.utils.data.DataLoader(train_images, batch_size=1, shuffle=True, num_workers=3)

srgb_profile = ImageCms.createProfile("sRGB")
lab_profile = ImageCms.createProfile("LAB", colorTemp=6500)

lab2rgb_transform = ImageCms.buildTransformFromOpenProfiles(lab_profile, srgb_profile, "LAB",
                                                                 "RGB")


for i, data in enumerate(trainloader):
    lightness_image, ground_truth_encoding, image_path = data

    print(image_path)

    if use_cuda:
        lightness_image, ground_truth_encoding \
            = Variable(lightness_image.cuda()), Variable(ground_truth_encoding.cuda())
    else:
        lightness_image, ground_truth_encoding = Variable(lightness_image), Variable(ground_truth_encoding)

    output = sabrina(lightness_image)

    output = output.permute(0,2,3,1)
    m = nn.Sigmoid()
    output = torch.squeeze(m(output).data).cpu().numpy()
    output_bins = get_annealed_means(output,TEMPERATURE)

    L = torch.squeeze(lightness_image.data).cpu().numpy()
    colorized = np.zeros((256,256,3))

    for i in range(256):
        for j in range(256):
            a,b = bin2ab[output_bins[i][j]]
            a = 255 if a > 255 else a
            b = 255 if b > 255 else b

            # Reversing PIL conversions to 0-255
            # colorized[i][j] = [L[i][j]*100/255,a-128,b-128]

            colorized[i][j] = [L[i][j],a,b]

    pil_lab_image = Image.fromarray(colorized.astype('uint8'), 'LAB')
    pil_rgb_image = ImageCms.applyTransform(pil_lab_image, lab2rgb_transform)

    pil_rgb_image.save('test_image.jpg', 'JPEG')

    break

