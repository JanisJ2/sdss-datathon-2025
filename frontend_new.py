import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from final import *

ctk.set_appearance_mode("System")  
ctk.set_default_color_theme("blue")

def backend_function(den, neighbourhood, beds, size, maint):
    print("Backend function called with:")
    print("Den:", den)
    print("Neighbourhood:", neighbourhood)
    print("Beds:", beds)
    print("Size:", size)
    print("Maint:", maint)

    input_data = {
        'DEN': [str(den)],
        'neighbourhood': [str(neighbourhood)],
        'beds': [int(beds)],
        'size': [str(size)],
        'maint': [float(maint)]
    }
    input_df = pd.DataFrame(input_data)
    prediction = final_pipeline.predict(input_df)
    print("Model Prediction:", prediction[0])
    return f"{int(prediction[0]):,}"

# Function called when the form is submitted.
def submit_form():
    den = den_var.get()
    neighbourhood = neighbourhood_combo.get()
    beds = beds_combo.get()
    size = size_combo.get()
    maint = maint_entry.get()

    try:
        beds_int = int(beds)
    except ValueError:
        messagebox.showerror("Invalid Input", "Beds must be an integer.")
        return

    try:
        maint_val = float(maint)
    except ValueError:
        messagebox.showerror("Invalid Input", "Maint must be a number.")
        return

    result = backend_function(den, neighbourhood, beds_int, size, maint_val)
    messagebox.showinfo("Prediction Result", f"Predicted Price: ~${result}")

# Create the main application window.
root = ctk.CTk()
root.title("Real Estate Price Prediction")
root.geometry("600x600")
root.resizable(True, True)
root.attributes("-fullscreen", True)

# Main frame with padding and rounded corners.
mainframe = ctk.CTkFrame(root, corner_radius=15)
mainframe.pack(padx=20, pady=20, fill="both", expand=True)

# Header Label.
header_label = ctk.CTkLabel(
    mainframe,
    text="Real Estate Price Prediction",
    font=ctk.CTkFont(family="Helvetica", size=24, weight="bold")
)
header_label.pack(pady=(20, 30))

# Den - Radio Buttons.
den_frame = ctk.CTkFrame(mainframe)
den_frame.pack(pady=(5, 15))
den_label = ctk.CTkLabel(den_frame, text="Den:", font=ctk.CTkFont(family="Helvetica", size=16))
den_label.grid(row=0, column=0, padx=10)
den_var = tk.StringVar(value="yes")
den_radio_yes = ctk.CTkRadioButton(
    den_frame, text="Yes", variable=den_var, value="yes",
    font=ctk.CTkFont(family="Helvetica", size=16)
)
den_radio_yes.grid(row=0, column=1, padx=10)
den_radio_no = ctk.CTkRadioButton(
    den_frame, text="No", variable=den_var, value="no",
    font=ctk.CTkFont(family="Helvetica", size=16)
)
den_radio_no.grid(row=0, column=2, padx=10)

# Neighbourhood Combobox.
neighbourhood_label = ctk.CTkLabel(
    mainframe, text="Neighbourhood:", font=ctk.CTkFont(family="Helvetica", size=16)
)
neighbourhood_label.pack(pady=(5, 0))
neighbourhood_combo = ctk.CTkComboBox(mainframe, values=[
    "St Lawrence-East Bayfront-The Islands",
    "Wellington Place",
    "Kensington-Chinatown",
    "Cabbagetown-South St.James Town",
    "Harbourfront-CityPlace",
    "Rosedale-Moore Park",
    "South Parkdale",
    "Regent Park",
    "Palmerston-Little Italy",
    "Annex",
    "North St.James Town",
    "Trinity-Bellwoods",
    "Fort York-Liberty Village",
    "Leaside-Bennington",
    "Moss Park",
    "Bay-Cloverhill",
    "Dovercourt Village",
    "Downtown Yonge East",
    "Yonge-Bay Corridor",
    "Church-Wellesley",
    "University",
    "West Queen West"
], width=350)
neighbourhood_combo.set("St Lawrence-East Bayfront-The Islands")
neighbourhood_combo.pack(pady=(0, 15))

# Beds - Dropdown instead of entry.
beds_label = ctk.CTkLabel(
    mainframe, text="Beds:", font=ctk.CTkFont(family="Helvetica", size=16)
)
beds_label.pack(pady=(5, 0))
beds_combo = ctk.CTkComboBox(mainframe, values=["0", "1", "2", "3"], width=100)
beds_combo.set("1")
beds_combo.pack(pady=(0, 15))

# Size Combobox.
size_label = ctk.CTkLabel(
    mainframe, text="Size:", font=ctk.CTkFont(family="Helvetica", size=16)
)
size_label.pack(pady=(5, 0))
size_combo = ctk.CTkComboBox(mainframe, values=[
    "0-499 sqft",
    "500-999 sqft",
    "1000-1499 sqft",
    "1500-1999 sqft",
    "2000-2499 sqft",
    "2500-2999 sqft",
    "3000-3499 sqft",
    "3500-3999 sqft",
    "4000+ sqft"
], width=200)
size_combo.set("0-499 sqft")
size_combo.pack(pady=(0, 15))

# Maint Entry.
maint_label = ctk.CTkLabel(
    mainframe, text="Maint:", font=ctk.CTkFont(family="Helvetica", size=16)
)
maint_label.pack(pady=(5, 0))
maint_entry = ctk.CTkEntry(mainframe, width=100, placeholder_text="Enter maintenance cost")
maint_entry.pack(pady=(0, 15))
maint_entry.insert(0, "0")

# Submit Button.
submit_button = ctk.CTkButton(mainframe, text="Submit", command=submit_form, width=150)
submit_button.pack(pady=(20, 20))

# Start the CustomTkinter event loop.
root.mainloop()
