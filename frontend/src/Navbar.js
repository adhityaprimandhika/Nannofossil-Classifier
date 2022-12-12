import Navbar from "react-bootstrap/Navbar";
import Container from "react-bootstrap/Container";

export default function NavBar() {
  return (
    <Navbar bg="dark" variant="dark" fixed="top">
      <Container className="navbar-container">
        <Navbar.Brand href="#">Nannofossil Classifier</Navbar.Brand>
      </Container>
    </Navbar>
  );
}
