import json

def process_json_with_ids(input_file, output_file):
    """
    Loads JSON data, converts it to a dictionary with IDs as keys, sums the total occurrences,
    and calculates the percentage for each ID.
    
    :param input_file: Path to the input JSON file
    :param output_file: Path to the output JSON file
    """
    try:
        # Read the JSON data from the file
        with open(input_file, 'r') as file:
            data = json.load(file)
        
        # Create a new dictionary with IDs as keys
        id_dict = {}
        total_occurrences = 0
        
        for category, objects in data.items():
            for obj in objects:
                # Add the object to the new dictionary using its 'id' as the key
                obj_id = obj["id"]
                id_dict[obj_id] = obj
                
                # Accumulate the total occurrences
                total_occurrences += obj["occurence"]
        
        # Calculate percentages for each ID
        for obj_id, obj in id_dict.items():
            obj["percentage"] = (obj["occurence"] / total_occurrences) * 100
        
        # Write the updated dictionary back to a file
        with open(output_file, 'w') as file:
            json.dump(dict(sorted(id_dict.items(), key=lambda item: item[1]["percentage"], reverse=True)), file, indent=4)

        print(f"Processed data written to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file = "data/annotations_with_ids.json"  # Replace with your input JSON file path (with IDs added)
output_file = "data/annotations_dict.json"  # Replace with your desired output JSON file path

process_json_with_ids(input_file, output_file)
