def process(image_file, label_file, output_file, samples):
    input_images = open(image_file, "rb")
    input_labels = open(label_file, "rb")
    output = open(output_file, "w")
    
    input_images.read(16)
    input_labels.read(8)
    images = []
    
    for i in range(samples):
        if i != 0 and i % 1000 == 0:
            print("Images Processed: {}/{}".format(i, samples))
        image = [ord(input_labels.read(1))]
        
        for j in range(28 * 28):
            image.append(ord(input_images.read(1)))
        
        images.append(image)
    
    for image in images:
        output.write(",".join(str(pixel) for pixel in image) + "\n")
    print("It seems that it is over\n")
    input_images.close()
    input_labels.close()
    output.close()


if __name__ == "__main__":
    print()
    # process("train-images.idx3-ubyte",
    #         "train-labels.idx1-ubyte", "mnist-train-data.csv", 60000)
    process("t10k-images.idx3-ubyte",
            "t10k-labels.idx1-ubyte", "mnist-test-data.csv", 10000)
    print("Finished")
