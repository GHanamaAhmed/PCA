import tkinter
import tkinter.messagebox
from tkinter.filedialog import askopenfilename
import customtkinter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from pandas.api.types import is_string_dtype
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
         # configure window
        self.title("CustomTkinter Dataset Visualizer")
        self.geometry(f"{1100}x{680}")

        # configure grid layout
        self.grid_columnconfigure(1, weight=1)  # Full width for textbox
        self.grid_rowconfigure(0, weight=1)  # Give weight to textbox row

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Dataset Visualizer",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Upload dataset button
        self.upload_button = customtkinter.CTkButton(self.sidebar_frame, text="Upload Dataset", command=self.upload_dataset)
        self.upload_button.grid(row=1, column=0, padx=20, pady=10)

        # Add buttons for visualization
        self.sidebar_button_scatter = customtkinter.CTkButton(self.sidebar_frame, text="Scatter Plot", command=self.plot_scatter, state="disabled")
        self.sidebar_button_scatter.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_pairplot = customtkinter.CTkButton(self.sidebar_frame, text="Pair Plot", command=self.plot_pairplot, state="disabled")
        self.sidebar_button_pairplot.grid(row=3, column=0, padx=20, pady=10)

        self.sidebar_button_pca = customtkinter.CTkButton(self.sidebar_frame, text="PCA Visualization", command=self.plot_pca, state="disabled")
        self.sidebar_button_pca.grid(row=4, column=0, padx=20, pady=10)

        # Appearance controls in sidebar
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=6, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=7, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 20))

        # create main content area (textbox) - now full width
        self.textbox = customtkinter.CTkTextbox(self)
        self.textbox.grid(row=0, column=1, padx=(20, 20), pady=(20, 0), sticky="nsew")

        # create bottom frame (formerly right frame) - no empty space
        self.bottom_frame = customtkinter.CTkFrame(self)
        self.bottom_frame.grid(row=1, column=1, padx=(20, 20), pady=(0, 20), sticky="ew")  # Changed to "ew" to prevent vertical expansion

        # Configure bottom frame grid
        self.bottom_frame.grid_columnconfigure((0, 1), weight=1)  # Equal width columns
        

        # Create left and right frames inside the bottom frame
        self.left_bottom_frame = customtkinter.CTkFrame(self.bottom_frame)
        self.left_bottom_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.right_bottom_frame = customtkinter.CTkFrame(self.bottom_frame)
        self.right_bottom_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Configure grid weights for left_bottom_frame
        self.left_bottom_frame.grid_columnconfigure(0, weight=1)
        self.left_bottom_frame.grid_columnconfigure(1, weight=1)

        # set default values
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        self.textbox.insert("0.0", "Please upload a dataset to begin visualizations.\n")

        # Initialize dataset variable
        self.df = None
        self.scatter_columns = []
        self.pca_components = None
        
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def upload_dataset(self):
        filepath = askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filepath:
            try:
                self.df = pd.read_csv(filepath)
                self.textbox.delete("0.0", "end")  # Clear existing text
                self.textbox.insert("0.0", f"Dataset loaded successfully from: {filepath}\n")
                self.textbox.insert("end", self.df.to_string(index=False))  # Display first 10 rows of the dataset
                self.enable_buttons()
                self.create_scatter_column_dropdown()
                self.create_pca_component_input()
            except Exception as e:
                tkinter.messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def enable_buttons(self):
        if self.df is not None:
            self.sidebar_button_scatter.configure(state="normal")
            self.sidebar_button_pairplot.configure(state="normal")
            self.sidebar_button_pca.configure(state="normal")

    def create_scatter_column_dropdown(self):
        # Remove old dropdown if exists
        if hasattr(self, 'scatter_column1_dropdown'):
            self.scatter_column1_dropdown.destroy()
            self.scatter_column2_dropdown.destroy()

        columns = self.df.columns.tolist()
        columns = [col for col in columns if self.df[col].dtype in [np.float64, np.int64]]

        self.scatter_columns = columns

        self.scatter_column1_label = customtkinter.CTkLabel(self.left_bottom_frame, text="X-axis:")
        self.scatter_column1_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        self.scatter_column1_dropdown = customtkinter.CTkOptionMenu(self.left_bottom_frame, values=self.scatter_columns)
        self.scatter_column1_dropdown.grid(row=3, column=0, padx=10, pady=5, sticky="w")

        self.scatter_column2_label = customtkinter.CTkLabel(self.left_bottom_frame, text="Y-axis:")
        self.scatter_column2_label.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        self.scatter_column2_dropdown = customtkinter.CTkOptionMenu(self.left_bottom_frame, values=self.scatter_columns)
        self.scatter_column2_dropdown.grid(row=3, column=1, padx=10, pady=5, sticky="w")

    def create_pca_component_input(self):
        # Remove old PCA input if exists
        if hasattr(self, 'pca_components_label'):
            self.pca_components_label.destroy()
        if hasattr(self, 'pca_components_input'):
            self.pca_components_input.destroy()
        if hasattr(self, 'pca_components_types_label'):
            self.pca_components_types_label.destroy()
        if hasattr(self, 'pca_components_types_dropDown'):
            self.pca_components_types_dropDown.destroy()
        if hasattr(self, 'feature_checkboxes_frame'):
            self.feature_checkboxes_frame.destroy()

        # Label for PCA Components
        self.pca_components_label = customtkinter.CTkLabel(self.left_bottom_frame, text="Number of Principal Components:")
        self.pca_components_label.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # Entry for Number of Components
        self.pca_components_input = customtkinter.CTkEntry(self.left_bottom_frame)
        self.pca_components_input.grid(row=5, column=0, padx=10, pady=5, sticky="w")

        self.pca_components_types_label = customtkinter.CTkLabel(self.left_bottom_frame, text="Categories filed:")
        self.pca_components_types_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        columns = self.df.columns.tolist()
        self.scatter_columns_str = columns
        columns = [col for col in columns if is_string_dtype(self.df[col])]

        self.pca_components_types_dropDown = customtkinter.CTkOptionMenu(self.left_bottom_frame, values=columns)
        self.pca_components_types_dropDown.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        # Frame for Checkboxes
        self.feature_checkboxes_frame = customtkinter.CTkFrame(self.left_bottom_frame)
        self.feature_checkboxes_frame.grid(row=6, column=0, padx=10, pady=5, sticky="w")
        
        # Dynamically create checkboxes for numerical features
        self.feature_checkboxes = {}
        feature_label = customtkinter.CTkLabel(self.feature_checkboxes_frame, text="Select Features for PCA:")
        feature_label.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        numerical_columns = [col for col in self.df.columns if self.df[col].dtype in [np.float64, np.int64]]
        for i, column in enumerate(numerical_columns):
            var = tkinter.BooleanVar()
            checkbox = customtkinter.CTkCheckBox(self.feature_checkboxes_frame, text=column, variable=var)
            checkbox.grid(row=(i // 2) + 1, column=i % 2, padx=10, pady=5, sticky="w")
            self.feature_checkboxes[column] = var

    def plot_scatter(self):
        if self.df is not None:
            x_column = self.scatter_column1_dropdown.get()
            y_column = self.scatter_column2_dropdown.get()

            if x_column and y_column:
                # Clear previous plot
                for widget in self.right_bottom_frame.winfo_children():
                    widget.destroy()

                # Create a matplotlib figure
                fig = Figure(figsize=(4, 4), dpi=100)
                ax = fig.add_subplot(111)
                for types in self.df[self.pca_components_types_dropDown.get()].unique():
                    subset=self.df[self.df[self.pca_components_types_dropDown.get()]==types]
                    ax.scatter(subset[x_column], subset[y_column],label=types)
                ax.set_title(f"Scatter Plot: {x_column} vs {y_column}")
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)

                # Embed the figure in Tkinter
                canvas = FigureCanvasTkAgg(fig, master=self.right_bottom_frame)
                canvas_widget = canvas.get_tk_widget()
                canvas_widget.pack(fill="both", expand=True)
                canvas.draw()


    def plot_pairplot(self):
        if self.df is not None:
            sns.pairplot(self.df, hue=self.pca_components_types_dropDown.get(), diag_kind="hist", height=2.5)
            plt.suptitle("Pairwise Feature Relationships", y=1.02, fontsize=16)
            plt.show()

    def plot_pca(self):
        if self.df is not None:
            # Clear previous plot
            for widget in self.right_bottom_frame.winfo_children():
                widget.destroy()

            selected_features = [col for col, var in self.feature_checkboxes.items() if var.get()]
            if not selected_features:
                tkinter.messagebox.showwarning("Warning", "Please select at least one feature for PCA.")
                return
            n_components = int(self.pca_components_input.get()) if self.pca_components_input.get() else 2
            features = selected_features
            x = self.df[features]
            # Handle missing values: Drop rows with NaN values
            x = x.dropna()
            if x.empty:
                tkinter.messagebox.showwarning("Warning", "All selected rows have missing values. Please check your data.")
                return
            
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
    
            pca = PCA(n_components=n_components)
            x_pca = pca.fit_transform(x_scaled)

            pca_df = pd.DataFrame(x_pca, columns=[f"PC{i+1}" for i in range(n_components)])
            pca_df[self.pca_components_types_dropDown.get()]=  self.df[self.pca_components_types_dropDown.get()]
            # Create a matplotlib figure
            fig = Figure(figsize=(4, 4), dpi=100)
            ax = fig.add_subplot(111)
            for spices in pca_df[self.pca_components_types_dropDown.get()].unique():
                subset=pca_df[pca_df[self.pca_components_types_dropDown.get()]==spices]
                ax.scatter(subset["PC1"],subset["PC2"],label=spices)
            ax.set_title(f"PCA with {n_components} Components")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            # Embed the figure in Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.right_bottom_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill="both", expand=True)
            canvas.draw()


if __name__ == "__main__":
    app = App()
    app.mainloop()
