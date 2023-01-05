import "./App.css";
import Home from "./pages/Home";
import NavBar from "./components/Navbar";
import Predictor from "./pages/Predictor";
import ErrorPage from "./pages/ErrorPage";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

function App() {
  const router = createBrowserRouter([
    {
      path: "/",
      element: <Home />,
      errorElement: <ErrorPage />,
    },
    {
      path: "/classification",
      element: <Predictor />,
    },
  ]);
  return (
    <div className="App">
      <NavBar />
      <RouterProvider router={router} />
    </div>
  );
}

export default App;
