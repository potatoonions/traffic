import tkinter as tk
from tkinter import ttk, messagebox
import requests
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class StockGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Market Prediction System")
        self.root.geometry("1200x800")
        
        # API configuration
        self.api_key = "ZTJL216JZCLZE26O"
        self.base_url = "http://127.0.0.1:8000"
        
        # Create main frames
        self.create_frames()
        self.create_widgets()
        
    def create_frames(self):
        # Search frame
        self.search_frame = ttk.LabelFrame(self.root, text="Company Search", padding="10")
        self.search_frame.pack(fill="x", padx=10, pady=5)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(self.root, text="Search Results", padding="10")
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Stock details frame
        self.stock_frame = ttk.LabelFrame(self.root, text="Stock Details", padding="10")
        self.stock_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
    def create_widgets(self):
        # Search widgets
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(self.search_frame, textvariable=self.search_var, width=50)
        search_entry.pack(side="left", padx=5, pady=5)
        
        search_button = ttk.Button(self.search_frame, text="Search", command=self.search_companies)
        search_button.pack(side="left", padx=5, pady=5)
        
        # Results treeview
        self.results_tree = ttk.Treeview(self.results_frame, columns=("Symbol", "Name", "Type", "Region", "Score"), show="headings")
        self.results_tree.heading("Symbol", text="Symbol")
        self.results_tree.heading("Name", text="Name")
        self.results_tree.heading("Type", text="Type")
        self.results_tree.heading("Region", text="Region")
        self.results_tree.heading("Score", text="Score")
        
        self.results_tree.column("Symbol", width=100)
        self.results_tree.column("Name", width=300)
        self.results_tree.column("Type", width=100)
        self.results_tree.column("Region", width=150)
        self.results_tree.column("Score", width=100)
        
        self.results_tree.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Bind double-click event
        self.results_tree.bind("<<TreeviewSelect>>", self.on_company_select)
        
        # Loading indicator
        self.loading_label = ttk.Label(self.results_frame, text="Searching...")
        self.loading_label.pack(pady=5)
        self.loading_label.pack_forget()  # Hide initially
        
        # Stock details widgets
        self.symbol_label = ttk.Label(self.stock_frame, text="Symbol:")
        self.symbol_label.pack(anchor="w", padx=5, pady=2)
        
        self.name_label = ttk.Label(self.stock_frame, text="Name:")
        self.name_label.pack(anchor="w", padx=5, pady=2)
        
        self.price_label = ttk.Label(self.stock_frame, text="Current Price:")
        self.price_label.pack(anchor="w", padx=5, pady=2)
        
        self.predict_button = ttk.Button(self.stock_frame, text="Predict Price", command=self.predict_price)
        self.predict_button.pack(pady=10)
        
        # Create a figure for plotting
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.stock_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
    def search_companies(self):
        query = self.search_var.get()
        if not query:
            messagebox.showwarning("Warning", "Please enter a search term")
            return
            
        # Show loading indicator
        self.loading_label.pack(pady=5)
        self.root.update_idletasks()
        
        try:
            url = f"{self.base_url}/search-company?query={query}"
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            results = response.json()
            
            # Clear existing items
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # Insert new items
            if results:
                for company in results:
                    self.results_tree.insert("", "end", values=(
                        company['symbol'],
                        company['name'],
                        company['type'],
                        company['region'],
                        company['matchScore']
                    ))
            else:
                messagebox.showinfo("No Results", "No companies found matching your search")
            
        except requests.exceptions.HTTPError as http_err:
            messagebox.showerror("Error", f"HTTP error occurred: {http_err}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to search companies: {str(e)}")
        finally:
            self.loading_label.pack_forget()  # Hide loading indicator
            
    def on_company_select(self, event):
        selected_item = self.results_tree.selection()
        if not selected_item:
            return
            
        item = self.results_tree.item(selected_item[0])
        symbol = item['values'][0]
        
        # Update company details
        self.symbol_label.config(text=f"Symbol: {symbol}")
        self.name_label.config(text=f"Name: {item['values'][1]}")
        
        # Fetch current price
        try:
            url = f"{self.base_url}/predict-stock"
            response = requests.post(url, json={"symbol": symbol})
            response.raise_for_status()
            result = response.json()
            
            self.price_label.config(text=f"Current Price: ${result['predicted_price']:.2f}")
            
            # Update chart
            self.update_chart(symbol)
            
        except requests.exceptions.HTTPError as http_err:
            messagebox.showerror("Error", f"HTTP error occurred: {http_err}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch stock details: {str(e)}")

    def predict_price(self):
        selected_item = self.results_tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a company first")
            return
            
        item = self.results_tree.item(selected_item[0])
        symbol = item['values'][0]
        
        try:
            url = f"{self.base_url}/predict-stock"
            response = requests.post(url, json={"symbol": symbol})
            response.raise_for_status()
            result = response.json()
            
            messagebox.showinfo("Prediction", 
                f"Predicted Price for {symbol}: ${result['predicted_price']:.2f}\n"
                f"Prediction Date: {result['prediction_date']}"
            )
            
            # Update chart
            self.update_chart(symbol)
            
        except requests.exceptions.HTTPError as http_err:
            messagebox.showerror("Error", f"HTTP error occurred: {http_err}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict price: {str(e)}")
            
    def update_chart(self, symbol):
        try:
            # Fetch historical data
            url = f"{self.base_url}/predict-stock"
            response = requests.post(url, json={"symbol": symbol})
            response.raise_for_status()
            result = response.json()
            
            # Clear previous plot
            self.ax.clear()
            
            # Plot historical data
            dates = [datetime.strptime(date, "%Y-%m-%d") for date in result['historical_dates']]
            prices = result['historical_prices']
            
            self.ax.plot(dates, prices, label="Historical Prices")
            
            # Add predicted price
            if 'prediction_date' in result:
                pred_date = datetime.strptime(result['prediction_date'], "%Y-%m-%d")
                self.ax.plot(pred_date, result['predicted_price'], 'ro', label="Predicted Price")
            
            self.ax.set_title(f"{symbol} Stock Prices")
            self.ax.set_xlabel("Date")
            self.ax.set_ylabel("Price")
            self.ax.legend()
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Update canvas
            self.canvas.draw()
            
        except requests.exceptions.HTTPError as http_err:
            messagebox.showerror("Error", f"HTTP error occurred: {http_err}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update chart: {str(e)}")

def main():
    root = tk.Tk()
    app = StockGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
