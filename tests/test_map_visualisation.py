import pytest
from unittest.mock import patch
from src.visualization.map_visualisation import Visualisation


class TestVisualisation:

    def test_init(self):
        vis = Visualisation()
        assert vis.clusters == []
        assert vis.markers == []
        assert vis.points == []

    def test_draw_cluster(self):
        vis = Visualisation()
        points = [[45.0, 4.0], [45.1, 4.1], [45.2, 4.0]]
        with patch('folium.Polygon') as mock_polygon:
            vis.draw_cluster(points, color='red')
            mock_polygon.assert_called_with(
                locations=points,
                color='red',
                fill=True,
                fill_color='blue',
                weight=2,
                opacity=0.7,
                fill_opacity=0.3)
            assert len(vis.clusters) == 1

    def test_draw_point_list(self):
        vis = Visualisation()
        point = [45.0, 4.0]
        with patch('folium.Marker') as mock_marker:
            vis.draw_point(point)
            mock_marker.assert_called_once()
            args, kwargs = mock_marker.call_args
            assert kwargs['location'] == [45.0, 4.0]
            assert 'popup' in kwargs
            assert len(vis.markers) == 1

    def test_draw_point_dict(self):
        vis = Visualisation()
        point = {'latitude': 45.0, 'longitude': 4.0, 'title': 'Test Point'}
        with patch('folium.Marker') as mock_marker:
            vis.draw_point(point)
            args, kwargs = mock_marker.call_args
            assert kwargs['location'] == [45.0, 4.0]
            assert kwargs['popup'] == 'Test Point'

    def test_draw_point_object(self):
        vis = Visualisation()

        class PointObj:
            latitude = 45.0
            longitude = 4.0
            title = 'Object Point'

        point = PointObj()
        with patch('folium.Marker') as mock_marker:
            vis.draw_point(point)
            args, kwargs = mock_marker.call_args
            assert kwargs['location'] == [45.0, 4.0]
            assert kwargs['popup'] == 'Object Point'

    def test_draw_point_invalid(self):
        vis = Visualisation()
        with pytest.raises(TypeError):
            vis.draw_point("invalid")

    @patch('src.visualization.map_visualisation.folium.Map')
    def test_create_map(self, mock_map_cls):
        vis = Visualisation()
        mock_map_instance = mock_map_cls.return_value

        # Add some data
        vis.draw_point([45.0, 4.0])
        vis.draw_cluster([[45.0, 4.0], [45.1, 4.1]])

        # Call create_map
        vis.create_map("test_output.html", bounds=[(44, 3), (46, 5)])

        assert mock_map_cls.called
        mock_map_instance.save.assert_called_with("test_output.html")

    @patch('src.visualization.map_visualisation.folium.Map')
    def test_create_map_no_bounds(self, mock_map_cls):
        vis = Visualisation()
        vis.create_map("test_output.html")
        mock_map_cls.assert_called()
