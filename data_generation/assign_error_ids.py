import json
import uuid

def assign_unique_ids(input_file, output_file):
    """
    Reads a JSON file, assigns unique IDs to each object, and writes the updated JSON to an output file.
    
    :param input_file: Path to the input JSON file
    :param output_file: Path to the output JSON file
    """
    try:
        # Read the JSON data from the file
        with open(input_file, 'r') as file:
            data = json.load(file)
        
        # Iterate over each key and its list of objects
        for category, objects in data.items():
            for index, obj in enumerate(objects):
                # Generate a unique ID (can be replaced with other schemes if needed)
                category = category.replace(" ", "_")
                unique_id = f"{category}-{index+1}-{uuid.uuid4().hex[:8]}"
                obj["id"] = unique_id

        # Write the updated JSON data back to the file
        with open(output_file, 'w') as file:
            json.dump(data, file, indent=4)

        print(f"Unique IDs assigned and written to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file = "data/annotations.json"  # Replace with your input JSON file path
output_file = "data/annotations_with_ids.json"  # Replace with your desired output JSON file path

assign_unique_ids(input_file, output_file)
