from main_tui import XOTApp

if __name__ == "__main__":
    try:
        app = XOTApp()
        app.run()
    except Exception as err:
        raise err