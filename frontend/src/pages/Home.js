import Container from "react-bootstrap/Container";
import Button from "react-bootstrap/Button";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import { Link } from "react-router-dom";

export default function Home() {
  return (
    <Container className="home">
      <Row>
        <Col className="col-home">
          <h1>Let's classify some nannofossil</h1>
          <h6>*For now only for Discoaster Genus</h6>
          <Link to="/classification">
            <Button variant="custom" id="btn-home">
              Start Classify
            </Button>
          </Link>
        </Col>
      </Row>
    </Container>
  );
}
